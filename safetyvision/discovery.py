"""Find DVR / NVR / XVR recorders and IP cameras on the network.

Discovery (no login needed):
  * ONVIF WS-Discovery multicast        - almost every IP camera / NVR
  * Dahua DHIP search (UDP 37810)        - Dahua, CP Plus, Amcrest ... (model, serial, MAC)
  * Hikvision SADP (UDP 37020)           - Hikvision, HiLook, HiWatch  (model, serial, channel counts)
  * TCP scan of camera ports on local subnets + extra ranges (works across routers, where multicast does not)
  * identification from HTTP / RTSP server banners and the MAC vendor (ARP table)

After the user enters credentials, detect_channels() asks the device how many channels it has
(Dahua CGI, Hikvision ISAPI, ONVIF media profiles) and falls back to probing RTSP channels.
"""
import base64
import concurrent.futures as cf
import datetime
import hashlib
import ipaddress
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET

CAMERA_PORTS = [554, 37777, 8000, 80, 8899, 34567, 5000, 8080]
PORT_HINTS = {37777: "dahua", 8000: "hikvision", 34567: "xiongmai", 8899: "onvif"}
OUI = {  # first 3 bytes of the MAC address -> vendor (display hint only)
    **{p: "Dahua" for p in ("3C:EF:8C", "4C:11:BF", "90:02:A9", "E0:50:8B", "14:A7:8B", "38:AF:29", "BC:32:5F",
                            "A0:BD:1D", "9C:14:63", "E4:24:6C", "E0:2E:FE")},
    **{p: "Hikvision" for p in ("44:19:B6", "4C:BD:8F", "54:C4:15", "BC:AD:28", "C0:56:E3", "28:57:BE", "A4:14:37",
                                "18:68:CB", "8C:E7:48", "E0:CA:3C", "98:DF:82", "44:47:CC", "24:28:FD", "58:03:FB",
                                "80:BE:AF", "C4:2F:90", "64:DB:8B", "10:12:FB", "68:6D:BC")},
    **{p: "Uniview" for p in ("48:EA:63", "6C:F1:7E")},
}
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# ============================================================================ helpers
def local_interfaces() -> list[tuple[str, ipaddress.IPv4Network]]:
    """(local ip, network) for every connected IPv4 interface (link-local, loopback and VPN /32s skipped)."""
    import psutil
    out = []
    stats = psutil.net_if_stats()
    for name, addrs in psutil.net_if_addrs().items():
        if name in stats and not stats[name].isup:
            continue
        for a in addrs:
            if a.family != socket.AF_INET or not a.netmask:
                continue
            ip = ipaddress.IPv4Address(a.address)
            if ip.is_loopback or ip.is_link_local:
                continue
            net = ipaddress.IPv4Network(f"{a.address}/{a.netmask}", strict=False)
            if net.prefixlen >= 31:
                continue
            if net.prefixlen < 24:  # large networks: scan the /24 around this PC
                net = ipaddress.IPv4Network(f"{a.address}/24", strict=False)
            out.append((a.address, net))
    return out


def parse_ranges(text: str) -> list[ipaddress.IPv4Address]:
    """'192.168.1.0/24, 10.0.0.10-10.0.0.60, 172.16.5.9' -> addresses (max 4096)"""
    hosts = []
    for part in re.split(r"[,\s;]+", text or ""):
        if not part:
            continue
        try:
            if "/" in part:
                hosts += list(ipaddress.IPv4Network(part, strict=False).hosts())
            elif "-" in part:
                a, b = part.split("-", 1)
                a = ipaddress.IPv4Address(a)
                b = ipaddress.IPv4Address(b) if "." in b else ipaddress.IPv4Address(str(a).rsplit(".", 1)[0] + "." + b)
                hosts += [ipaddress.IPv4Address(i) for i in range(int(a), int(b) + 1)]
            else:
                hosts.append(ipaddress.IPv4Address(part))
        except ValueError:
            continue
    return list(dict.fromkeys(hosts))[:4096]


def arp_table() -> dict[str, str]:
    try:
        out = subprocess.run(["arp", "-a"], capture_output=True, text=True, creationflags=NO_WINDOW).stdout
    except Exception:
        return {}
    table = {}
    for ip, mac in re.findall(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F]{2}(?:[-:][0-9a-fA-F]{2}){5})", out):
        mac = mac.upper().replace("-", ":")
        if not mac.startswith(("FF:FF", "01:00:5E")):
            table[ip] = mac
    return table


def vendor(mac: str) -> str:
    return OUI.get((mac or "")[:8].upper(), "")


# ============================================================================ discovery protocols
def _xml_text(root, tag):
    el = root.find(f".//{{*}}{tag}")
    return el.text.strip() if el is not None and el.text else ""


def ws_discovery(ifaces, timeout=3.0) -> dict:
    probe = ('<?xml version="1.0" encoding="UTF-8"?><e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
             'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
             'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
             'xmlns:dn="http://www.onvif.org/ver10/network/wsdl"><e:Header>'
             f'<w:MessageID>uuid:{uuid.uuid4()}</w:MessageID>'
             '<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>'
             '<w:Action e:mustUnderstand="true">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action>'
             '</e:Header><e:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>')
    found = {}
    socks = []
    for lip, _ in ifaces:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(lip))
            s.bind((lip, 0))
            s.settimeout(0.3)
            s.sendto(probe.encode(), ("239.255.255.250", 3702))
            socks.append(s)
        except OSError:
            pass
    end = time.time() + timeout
    while time.time() < end and socks:
        for s in socks:
            try:
                data, addr = s.recvfrom(65535)
            except (socket.timeout, OSError):
                continue
            try:
                parsed = parse_ws_probe_match(data.decode("utf-8", "ignore"))
            except ET.ParseError:
                continue
            found.setdefault(addr[0], {}).update(parsed)
    for s in socks:
        s.close()
    return found


def parse_ws_probe_match(xml_text: str) -> dict:
    root = ET.fromstring(xml_text)
    xaddrs = _xml_text(root, "XAddrs").split()
    scopes = _xml_text(root, "Scopes").split()
    info = {"onvif_xaddr": next((x for x in xaddrs if "://" in x), ""), "methods": {"ONVIF"}}
    for sc in scopes:
        m = re.match(r"onvif://www\.onvif\.org/(name|hardware|type|location)/(.+)", sc)
        if m:
            val = requests_unquote(m.group(2))
            key = {"name": "name", "hardware": "model", "location": "location_hint"}.get(m.group(1))
            if key:
                info[key] = val
            elif m.group(1) == "type" and val.lower() in ("nvr", "dvr", "network_video_recorder"):
                info["kind"] = "Recorder"
    return info


def requests_unquote(s):
    from urllib.parse import unquote
    return unquote(s).replace("_", " ")


def dahua_search(ifaces, timeout=3.0) -> dict:
    body = json.dumps({"method": "DHDiscover.search", "params": {"mac": "", "uni": 1}}).encode()
    pkt = b"\x20\x00\x00\x00DHIP" + b"\x00" * 8 + struct.pack("<II", len(body), len(body)) + body
    found, socks = {}, []
    for lip, net in ifaces:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.bind((lip, 0))
            s.settimeout(0.3)
            for dst in (str(net.broadcast_address), "255.255.255.255"):
                s.sendto(pkt, (dst, 37810))
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(lip))
            s.sendto(pkt, ("239.255.255.251", 37810))
            socks.append(s)
        except OSError:
            pass
    end = time.time() + timeout
    while time.time() < end and socks:
        for s in socks:
            try:
                data, addr = s.recvfrom(65535)
            except (socket.timeout, OSError):
                continue
            info = parse_dhip(data)
            if info:
                found.setdefault(info.pop("ip", addr[0]), {}).update(info)
    for s in socks:
        s.close()
    return found


def parse_dhip(data: bytes) -> dict | None:
    i = data.find(b"{")
    if i < 0:
        return None
    try:
        msg = json.loads(data[i:].decode("utf-8", "ignore").rstrip("\x00"))
    except json.JSONDecodeError:
        return None
    d = (msg.get("params") or {}).get("deviceInfo") or {}
    if not d:
        return None
    dtype = str(d.get("DeviceType", ""))
    info = {"brand": "dahua", "model": dtype, "serial": d.get("SerialNo", ""), "mac": str(d.get("Mac", "")).upper(),
            "name": d.get("MachineName", ""), "http_port": d.get("HttpPort"), "methods": {"Dahua search"},
            "kind": "Recorder" if re.search(r"(XVR|NVR|DVR|HCVR)", dtype, re.I) else "IP camera"}
    ip = ((d.get("IPv4Address") or {}).get("IPAddress"))
    if ip:
        info["ip"] = ip
    return info


def hik_sadp(ifaces, timeout=3.0) -> dict:
    probe = (f'<?xml version="1.0" encoding="utf-8"?><Probe><Uuid>{str(uuid.uuid4()).upper()}</Uuid>'
             '<Types>inquiry</Types></Probe>').encode()
    found, socks = {}, []
    for lip, _ in ifaces:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(lip))
            s.bind((lip, 0))
            s.settimeout(0.3)
            s.sendto(probe, ("239.255.255.250", 37020))
            socks.append(s)
        except OSError:
            pass
    end = time.time() + timeout
    while time.time() < end and socks:
        for s in socks:
            try:
                data, addr = s.recvfrom(65535)
            except (socket.timeout, OSError):
                continue
            info = parse_sadp(data.decode("utf-8", "ignore"))
            if info:
                found.setdefault(info.pop("ip", addr[0]), {}).update(info)
    for s in socks:
        s.close()
    return found


def parse_sadp(text: str) -> dict | None:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    if root.tag.split("}")[-1] != "ProbeMatch":
        return None
    g = lambda t: _xml_text(root, t)  # noqa: E731
    analog, digital = g("AnalogChannelNum"), g("DigitalChannelNum")
    desc = g("DeviceDescription") or g("DeviceType")
    info = {"brand": "hikvision", "model": desc, "serial": g("DeviceSN"), "mac": g("MAC").upper().replace("-", ":"),
            "http_port": int(g("HttpPort") or 80), "methods": {"Hikvision SADP"},
            "kind": "Recorder" if re.search(r"(NVR|DVR|HVR)", desc, re.I) or int(digital or 0) > 1 or int(analog or 0) > 1
            else "IP camera"}
    ch = int(analog or 0) + int(digital or 0)
    if ch:
        info["channels"] = ch
    if g("IPv4Address"):
        info["ip"] = g("IPv4Address")
    return info


def port_scan(hosts, ports=CAMERA_PORTS, timeout=0.35, progress=None, stop=None) -> dict:
    found = {}
    lock = threading.Lock()
    done = [0]

    def check(h):
        if stop and stop.is_set():
            return
        opened = []
        for p in ports:
            try:
                with socket.create_connection((str(h), p), timeout):
                    opened.append(p)
            except OSError:
                if p == 554 and not opened and p == ports[0]:
                    pass
        with lock:
            done[0] += 1
            if opened:
                found[str(h)] = opened
        if progress and done[0] % 16 == 0:
            progress(done[0], len(hosts))

    with cf.ThreadPoolExecutor(256) as ex:
        list(ex.map(check, hosts))
    return found


def identify(ip, ports) -> dict:
    """Guess brand / type from open ports and HTTP / RTSP server banners."""
    info = {"ports": sorted(ports), "methods": {"port scan"}}
    for p, brand in PORT_HINTS.items():
        if p in ports and brand != "onvif":
            info.setdefault("brand", brand)
    banner = ""
    for hp in (80, 8080):
        if hp in ports:
            try:
                import requests
                r = requests.get(f"http://{ip}:{hp}/", timeout=2, allow_redirects=True)
                banner += f" {r.headers.get('Server', '')} {' '.join(re.findall(r'<title>([^<]*)</title>', r.text, re.I))}"
                info["http_port"] = hp
            except Exception:
                pass
            break
    if 554 in ports:
        try:
            with socket.create_connection((ip, 554), 2) as s:
                s.sendall(f"OPTIONS rtsp://{ip}:554/ RTSP/1.0\r\nCSeq: 1\r\n\r\n".encode())
                resp = s.recv(1024).decode("utf-8", "ignore")
                m = re.search(r"Server:\s*([^\r\n]+)", resp)
                banner += f" {m.group(1) if m else ''}"
        except OSError:
            pass
    b = banner.lower()
    if any(k in b for k in ("dahua", "dh-", "webs", "cp plus", "amcrest")):
        info["brand"] = "dahua"
    elif any(k in b for k in ("hikvision", "hik", "app-webs", "dnvrs-webs", "davinci")):
        info["brand"] = "hikvision"
    elif "uniview" in b or "unv" in b:
        info["brand"] = "uniview"
    info["banner"] = banner.strip()[:80]
    if 554 in ports or 37777 in ports or 8000 in ports or 34567 in ports:
        info.setdefault("kind", "Recorder or camera")
    return info


def discover(extra_ranges="", known_hosts=(), progress=None, stop=None) -> list[dict]:
    """Run all discovery methods. progress(text, fraction)."""
    say = progress or (lambda *_: None)
    ifaces = local_interfaces()
    say(f"Searching {len(ifaces)} network(s) (ONVIF, Dahua, Hikvision) ...", 0.02)
    results: dict[str, dict] = {}

    def merge(found):
        for ip, info in found.items():
            cur = results.setdefault(ip, {"ip": ip, "methods": set()})
            methods = info.pop("methods", set())
            for k, v in info.items():
                if v not in (None, "", 0) and not cur.get(k):
                    cur[k] = v
            cur["methods"] |= set(methods)

    with cf.ThreadPoolExecutor(3) as ex:
        futs = [ex.submit(ws_discovery, ifaces), ex.submit(dahua_search, ifaces), ex.submit(hik_sadp, ifaces)]
        for f in futs:
            try:
                merge(f.result())
            except Exception:
                pass
    # TCP scan: local subnets + subnets of known recorders + extra ranges
    hosts = []
    for lip, net in ifaces:
        hosts += [h for h in net.hosts() if str(h) != lip]
    for h in known_hosts:
        try:
            hosts += list(ipaddress.IPv4Network(f"{h}/24", strict=False).hosts())
        except ValueError:
            pass
    hosts += parse_ranges(extra_ranges)
    local_ips = {lip for lip, _ in ifaces}
    hosts = [h for h in dict.fromkeys(hosts) if str(h) not in local_ips][:4096]
    say(f"Scanning {len(hosts)} addresses for camera ports ...", 0.1)
    scanned = port_scan(hosts, progress=lambda d, n: say(f"Scanning {d}/{n} addresses ...", 0.1 + 0.75 * d / n),
                        stop=stop)
    # only hosts that look like video devices (an RTSP / vendor port, or already found by a camera protocol)
    video_ports = {554, 37777, 8000, 34567, 8899}
    candidates = {ip: p for ip, p in scanned.items() if video_ports & set(p) or ip in results}
    say(f"Identifying {len(candidates)} device(s) ...", 0.87)
    with cf.ThreadPoolExecutor(16) as ex:
        ids = dict(zip(candidates, ex.map(lambda ip: identify(ip, candidates[ip]), candidates)))
    merge(ids)
    arp = arp_table()
    for ip, d in results.items():
        d.setdefault("ports", scanned.get(ip, []))
        if not d.get("mac") and ip in arp:
            d["mac"] = arp[ip]
        d["vendor"] = vendor(d.get("mac", "")) or ""
        if not d.get("brand") and d["vendor"]:
            d["brand"] = d["vendor"].lower()
        if not d.get("kind"):
            d["kind"] = "IP camera" if "ONVIF" in d["methods"] else "Unknown"
        d["methods"] = sorted(d["methods"])
    say(f"Found {len(results)} device(s)", 1.0)
    return sorted(results.values(), key=lambda d: tuple(int(x) for x in d["ip"].split(".")))


# ============================================================================ ONVIF client (for any IP camera)
def _wsse(user, pw):
    nonce = os.urandom(16)
    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + pw.encode()).digest()).decode()
    return ('<s:Header><Security s:mustUnderstand="1" xmlns="http://docs.oasis-open.org/wss/2004/01/'
            'oasis-200401-wss-wssecurity-secext-1.0.xsd"><UsernameToken><Username>' + user + '</Username>'
            '<Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0'
            f'#PasswordDigest">{digest}</Password><Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/'
            f'oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</Nonce>'
            '<Created xmlns="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd">'
            f'{created}</Created></UsernameToken></Security></s:Header>')


def _soap(url, body, user, pw, timeout=6):
    import requests
    env = ('<?xml version="1.0" encoding="UTF-8"?><s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" '
           'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:trt="http://www.onvif.org/ver10/media/wsdl" '
           'xmlns:tt="http://www.onvif.org/ver10/schema">' + _wsse(user, pw) + f"<s:Body>{body}</s:Body></s:Envelope>")
    r = requests.post(url, data=env.encode(), timeout=timeout,
                      headers={"Content-Type": "application/soap+xml; charset=utf-8"})
    if r.status_code == 401 or "NotAuthorized" in r.text:
        raise PermissionError("wrong username or password")
    r.raise_for_status()
    return ET.fromstring(r.content)


def onvif_streams(xaddr, user, pw, prefer="main") -> list[dict]:
    """[{source, profile, width, height, uri}] - one entry per video source (camera / channel)."""
    root = _soap(xaddr, "<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>", user, pw)
    media = _xml_text(root, "XAddr") or xaddr
    root = _soap(media, "<trt:GetProfiles/>", user, pw)
    by_source = {}
    for prof in root.iter():
        if prof.tag.split("}")[-1] != "Profiles":
            continue
        token = prof.get("token")
        src = prof.find(".//{*}VideoSourceConfiguration/{*}SourceToken")
        w, h = prof.find(".//{*}VideoEncoderConfiguration/{*}Resolution/{*}Width"), \
            prof.find(".//{*}VideoEncoderConfiguration/{*}Resolution/{*}Height")
        if token is None or src is None:
            continue
        px = (int(w.text) if w is not None else 0) * (int(h.text) if h is not None else 0)
        cur = by_source.get(src.text)
        better = cur is None or (px > cur["px"] if prefer == "main" else 0 < px < cur["px"])
        if better:
            by_source[src.text] = {"source": src.text, "profile": token, "px": px,
                                   "width": int(w.text) if w is not None else 0,
                                   "height": int(h.text) if h is not None else 0}
    out = []
    for s in by_source.values():
        body = ('<trt:GetStreamUri><trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream><tt:Transport>'
                f'<tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup><trt:ProfileToken>{s["profile"]}'
                '</trt:ProfileToken></trt:GetStreamUri>')
        uri = _xml_text(_soap(media, body, user, pw), "Uri")
        if uri:
            out.append({**s, "uri": uri})
    return out


# ============================================================================ channel detection (needs login)
def _digest_get(url, user, pw, timeout=5):
    import requests
    from requests.auth import HTTPBasicAuth, HTTPDigestAuth
    r = requests.get(url, auth=HTTPDigestAuth(user, pw), timeout=timeout)
    if r.status_code == 401:
        r = requests.get(url, auth=HTTPBasicAuth(user, pw), timeout=timeout)
    return r


def detect_channels(dev: dict, user: str, pw: str, stream="main") -> dict:
    """Returns {ok, brand, channels, onvif_streams?, message}."""
    ip, brand = dev["ip"], dev.get("brand", "")
    http = dev.get("http_port") or (80 if 80 in dev.get("ports", []) else 8080 if 8080 in dev.get("ports", []) else None)
    if dev.get("channels"):
        return {"ok": True, "brand": brand or "dahua", "channels": dev["channels"], "message": "reported by device"}
    if http and brand in ("dahua", ""):
        try:
            r = _digest_get(f"http://{ip}:{http}/cgi-bin/devVideoInput.cgi?action=getCollect", user, pw)
            m = re.search(r"result=(\d+)", r.text)
            if r.status_code == 200 and m:
                extra = {}
                for key, q in (("model", "getDeviceType"), ("serial", "getSerialNo")):
                    try:
                        mm = re.search(r"=(.+)", _digest_get(f"http://{ip}:{http}/cgi-bin/magicBox.cgi?action={q}",
                                                             user, pw).text)
                        extra[key] = mm.group(1).strip() if mm else ""
                    except Exception:
                        pass
                info = " ".join(v for v in (extra.get("model"), extra.get("serial")) if v)
                return {"ok": True, "brand": "dahua", "channels": int(m.group(1)), **extra,
                        "message": f"Dahua API {info}".strip()}
            if r.status_code == 401:
                return {"ok": False, "brand": "dahua", "message": "wrong username or password"}
        except Exception:
            pass
    if http and brand in ("hikvision", ""):
        try:
            n = 0
            for path, tag in (("/ISAPI/System/Video/inputs/channels", "VideoInputChannel"),
                              ("/ISAPI/ContentMgmt/InputProxy/channels", "InputProxyChannel")):
                r = _digest_get(f"http://{ip}:{http}{path}", user, pw)
                if r.status_code == 401:
                    return {"ok": False, "brand": "hikvision", "message": "wrong username or password"}
                if r.status_code == 200:
                    n += len(re.findall(f"<{tag}[ >]", r.text))
            if n:
                return {"ok": True, "brand": "hikvision", "channels": n, "message": "Hikvision ISAPI"}
        except Exception:
            pass
    if dev.get("onvif_xaddr"):
        try:
            streams = onvif_streams(dev["onvif_xaddr"], user, pw, stream)
            if streams:
                return {"ok": True, "brand": "onvif", "channels": len(streams), "onvif_streams": streams,
                        "message": "ONVIF"}
        except PermissionError as e:
            return {"ok": False, "brand": "onvif", "message": str(e)}
        except Exception:
            pass
    # last resort: open RTSP channels one by one with the brand's URL pattern
    if 554 in dev.get("ports", [554]):
        from .xvr import build_url, probe_source
        guess = brand if brand in ("dahua", "hikvision") else "dahua"
        rec = {"brand": guess, "host": ip, "port": 554, "username": user, "password": pw, "stream": "sub"}
        n, misses = 0, 0
        for ch in range(1, 65):
            ok, _ = probe_source(build_url(rec, ch), 8)
            if ok:
                n, misses = ch, 0
            else:
                misses += 1
                if misses >= (1 if n == 0 else 2):
                    break
        if n:
            return {"ok": True, "brand": guess, "channels": n, "message": "RTSP probe"}
        return {"ok": False, "brand": guess, "message": "no video - check username / password"}
    return {"ok": False, "brand": brand, "message": "not a video device (no RTSP)"}
