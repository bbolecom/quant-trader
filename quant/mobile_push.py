"""手机推送：支持 Bark(iOS) / Server酱(微信) / ntfy。

零额外依赖（仅标准库 urllib），可在本地和 GitHub Actions 运行。
密钥优先读环境变量（适配 GitHub Secrets），其次读配置内联值。

配置示例（写在各策略 config.json 的 "push_mobile" 节）：
{
  "push_mobile": {
    "enabled": true,
    "bark":      { "enabled": true,  "key_env": "BARK_KEY",        "key": "",
                   "server": "https://api.day.app", "group": "美股做空", "sound": "alarm" },
    "serverchan":{ "enabled": false, "key_env": "SERVERCHAN_KEY",  "key": "" },
    "ntfy":      { "enabled": false, "topic": "",                   "server": "https://ntfy.sh" }
  }
}

环境变量（任选其一渠道即可）：
  BARK_KEY=xxxxxxxx          # Bark App「服务器」页里的设备 key
  SERVERCHAN_KEY=SCTxxxx     # Server酱 SendKey
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request


def _post(url: str, data: bytes | None = None, headers: dict | None = None,
          timeout: int = 12) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, data=data, headers=headers or {},
                                     method="POST" if data is not None else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300, f"HTTP {resp.status}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _secret(node: dict) -> str:
    env_name = node.get("key_env") or ""
    val = os.environ.get(env_name, "") if env_name else ""
    return (val or str(node.get("key") or "")).strip()


def _push_bark(node: dict, title: str, body: str) -> tuple[bool, str]:
    key = _secret(node)
    if not key:
        return False, "缺少 Bark key（设 BARK_KEY 或 config.key）"
    server = str(node.get("server") or "https://api.day.app").rstrip("/")
    payload = {"title": title, "body": body, "device_key": key}
    if node.get("group"):
        payload["group"] = node["group"]
    if node.get("sound"):
        payload["sound"] = node["sound"]
    data = json.dumps(payload).encode("utf-8")
    ok, msg = _post(f"{server}/push", data=data,
                    headers={"Content-Type": "application/json; charset=utf-8"})
    if ok:
        return True, "bark ok"
    # 回退到 GET 路径式（老版本 Bark）
    t = urllib.parse.quote(title, safe="")
    b = urllib.parse.quote(body, safe="")
    ok2, msg2 = _post(f"{server}/{key}/{t}/{b}")
    return ok2, f"bark {'ok(get)' if ok2 else msg2}"


def _push_serverchan(node: dict, title: str, body: str) -> tuple[bool, str]:
    key = _secret(node)
    if not key:
        return False, "缺少 Server酱 SendKey"
    data = urllib.parse.urlencode({"title": title, "desp": body}).encode("utf-8")
    ok, msg = _post(f"https://sctapi.ftqq.com/{key}.send", data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"})
    return ok, f"serverchan {'ok' if ok else msg}"


def _push_ntfy(node: dict, title: str, body: str) -> tuple[bool, str]:
    topic = str(node.get("topic") or "").strip()
    if not topic:
        return False, "缺少 ntfy topic"
    server = str(node.get("server") or "https://ntfy.sh").rstrip("/")
    ok, msg = _post(f"{server}/{topic}", data=body.encode("utf-8"),
                    headers={"Title": title.encode("utf-8").decode("latin-1", "ignore")})
    return ok, f"ntfy {'ok' if ok else msg}"


def push_mobile(cfg: dict, title: str, body: str) -> list[str]:
    """按配置向所有启用渠道推送。返回结果日志行列表。"""
    if os.environ.get("QUANT_SKIP_MOBILE_PUSH") == "1":
        return []
    mc = (cfg or {}).get("push_mobile") or {}
    if not mc.get("enabled"):
        return []
    logs: list[str] = []
    handlers = {
        "bark": _push_bark,
        "serverchan": _push_serverchan,
        "ntfy": _push_ntfy,
    }
    for name, fn in handlers.items():
        node = mc.get(name) or {}
        if not node.get("enabled"):
            continue
        try:
            ok, msg = fn(node, title, body)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, str(e)
        line = f"[手机推送] {name}: {'✅' if ok else '⚠️ ' + msg}"
        logs.append(line)
        print(line, file=sys.stderr if not ok else sys.stdout)
    return logs
