# flake8: noqa: E501
import asyncio
import hashlib
import json
import random
import re
import time
from typing import Optional

try:
    import qrcode

    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

import httpx

# TODO: 实现稳定QQ登录


class QQLogin:

    def __init__(self, uin: int = 0, password: str = ""):
        self.uin = uin
        self.password = password
        self.token: Optional[str] = None
        self.logged_in = False
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://qzone.qq.com/",
            },
            follow_redirects=True,
        )
        self.cookies = {}
        self.qr_ticket = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def login(self) -> bool:
        try:
            if self.password:
                return await self._password_login()
            else:
                return await self._qr_login()
        except Exception as e:
            print(f"Login error: {e}")
            return False
        finally:
            await self.client.aclose()

    async def _qr_login(self) -> bool:
        print("\nPlease scan QR code with QQ...")

        qr_data = await self._get_qr_code()
        if not qr_data:
            print("Failed to get QR code")
            return False

        self._print_qr_code(qr_data)
        print("\nWaiting for scan...")

        max_attempts = 180
        attempt = 0

        while attempt < max_attempts and not self.logged_in:
            status = await self._check_qr_status()

            if status == "confirmed":
                token = await self._get_login_token()
                if token:
                    self.token = token
                    self.logged_in = True
                    print("\nLogin successful!")
                    return True
                else:
                    print("\nLogin failed: Unable to get token")
                    return False

            elif status == "expired":
                print("\nQR code expired, regenerating...")
                qr_data = await self._get_qr_code()
                if qr_data:
                    self._print_qr_code(qr_data)
                    print("\nPlease scan the new QR code...")
                    attempt = 0
                else:
                    print("Failed to regenerate QR code")
                    return False

            elif status == "cancelled":
                print("\nUser cancelled login")
                return False

            await asyncio.sleep(5)
            attempt += 1

        print("\nLogin timeout")
        return False

    async def _get_qr_code(self) -> Optional[str]:
        try:
            url = "https://ssl.ptlogin2.qq.com/ptqrshow"
            params = {
                "appid": "549000912",
                "e": "2",
                "l": "M",
                "s": "3",
                "d": "72",
                "v": "4",
                "t": str(random.random()),
                "daid": "5",
                "pt_3rd_aid": "0",
            }

            response = await self.client.get(url, params=params)

            for cookie in response.cookies.jar:
                if cookie.name == "qrsig":
                    self.cookies["qrsig"] = cookie.value
                    self.client.cookies.set(cookie.name, cookie.value)

            qrsig = self.cookies.get("qrsig", "")
            if qrsig:
                self.ptqrtoken = self._calculate_ptqrtoken(qrsig)
                qr_content = f"https://ptlogin2.qq.com/qr/{qrsig}"
                return qr_content

            return None

        except Exception as e:
            print(f"Failed to get QR code: {e}")
            return None

    def _calculate_ptqrtoken(self, qrsig: str) -> int:
        h = 0
        for char in qrsig:
            h += (h << 5) + ord(char)
        return 2147483647 & h

    async def _check_qr_status(self) -> str:
        try:
            qrsig = self.cookies.get("qrsig", "")
            if not qrsig:
                return "expired"

            ptqrtoken = getattr(self, "ptqrtoken", None)
            if not ptqrtoken:
                ptqrtoken = self._calculate_ptqrtoken(qrsig)
                self.ptqrtoken = ptqrtoken

            url = "https://ssl.ptlogin2.qq.com/ptqrlogin"
            timestamp_ms = int(time.time() * 1000)
            params = {
                "u1": "https://qzs.qq.com/qzone/v5/loginsucc.html",
                "ptqrtoken": str(ptqrtoken),
                "ptredirect": "0",
                "h": "1",
                "t": "1",
                "g": "1",
                "from_ui": "1",
                "ptlang": "2052",
                "action": f"0-0-{timestamp_ms}",
                "js_ver": "10291",
                "js_type": "1",
                "login_sig": "",
                "pt_uistyle": "40",
                "aid": "549000912",
                "daid": "5",
                "pt_3rd_aid": "0",
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://qzs.qq.com/qzone/v5/loginsucc.html",
            }

            response = await self.client.get(
                url, params=params, headers=headers, follow_redirects=False
            )
            text = response.text

            match = re.search(
                r"ptuiCB\('(\d+)','(\d+)','([^']*)','([^']*)','([^']*)'", text
            )
            if match:
                code = match.group(1)
                if code == "0":
                    redirect_url = match.group(3)
                    if redirect_url:
                        await self._follow_login_redirect(redirect_url)
                    return "confirmed"
                elif code == "66":
                    return "expired"
                elif code == "67":
                    return "pending"
                elif code == "7":
                    print(f"Parameter error: {text}")
                    return "expired"

            if "二维码已失效" in text or "已过期" in text:
                return "expired"

            if "用户取消" in text or "取消登录" in text:
                return "cancelled"

            return "pending"

        except Exception as e:
            print(f"Failed to check status: {e}")
            return "pending"

    async def _follow_login_redirect(self, redirect_url: str):
        try:
            response = await self.client.get(
                redirect_url, follow_redirects=True
            )
            for cookie in response.cookies.jar:
                self.cookies[cookie.name] = cookie.value
                self.client.cookies.set(cookie.name, cookie.value)
        except Exception as e:
            print(f"Failed to follow redirect: {e}")

    async def _get_login_token(self) -> Optional[str]:
        try:
            cookies = self.client.cookies
            skey = cookies.get("skey", "")
            p_skey = cookies.get("p_skey", "")
            uin = cookies.get("uin", "")

            if skey and uin:
                token_data = {
                    "uin": uin,
                    "skey": skey,
                    "p_skey": p_skey,
                }
                return json.dumps(token_data)

            return None

        except Exception as e:
            print(f"Failed to get token: {e}")
            return None

    def _print_qr_code(self, data: str):
        if QRCODE_AVAILABLE:
            try:
                qr = qrcode.QRCode(box_size=1, border=2)
                qr.add_data(data)
                qr.make(fit=True)
                print()
                qr.print_ascii(invert=True)
                print()
            except Exception as e:
                print(f"\nQR code data: {data}")
                print(f"Failed to display QR code: {e}")
        else:
            print(f"\nQR code URL: {data}")

    async def _password_login(self) -> bool:
        print(f"Logging in with password, UIN: {self.uin}")

        try:
            verify_params = await self._get_verify_params()
            if not verify_params:
                print("Failed to get login parameters")
                return False

            encrypted_password = await self._encrypt_password(
                self.password, verify_params
            )
            if not encrypted_password:
                print("Failed to encrypt password")
                return False

            login_result = await self._perform_password_login(
                verify_params, encrypted_password
            )

            if login_result:
                self.token = login_result.get("token")
                self.logged_in = True
                print("Login successful!")
                return True
            else:
                print("Login failed")
                return False

        except Exception as e:
            print(f"Password login error: {e}")
            return False

    async def _get_verify_params(self) -> Optional[dict]:
        try:
            url = "https://ssl.ptlogin2.qq.com/check"
            params = {
                "uin": str(self.uin),
                "appid": "549000912",
                "js_ver": "10291",
                "js_type": "1",
                "login_sig": "",
                "u1": "https://qzs.qq.com/qzone/v5/loginsucc.html",
                "r": str(random.random()),
            }

            return {
                "need_vc": "0",
                "rsa_key": "",
            }

        except Exception as e:
            print(f"Failed to get verify params: {e}")
            return None

    async def _encrypt_password(
        self, password: str, params: dict
    ) -> Optional[str]:
        try:
            rsa_key = params.get("rsa_key", "")
            if not rsa_key:
                return None

            return hashlib.md5(password.encode()).hexdigest()

        except Exception as e:
            print(f"Failed to encrypt password: {e}")
            return None

    async def _perform_password_login(
        self, verify_params: dict, encrypted_password: str
    ) -> Optional[dict]:
        try:
            url = "https://ssl.ptlogin2.qq.com/login"
            data = {
                "u": str(self.uin),
                "p": encrypted_password,
                "verifycode": verify_params.get("verifycode", ""),
                "webqq_type": "10",
                "remember_uin": "1",
                "login2qq": "1",
                "aid": "549000912",
                "u1": "https://qzs.qq.com/qzone/v5/loginsucc.html",
                "h": "1",
                "ptredirect": "0",
                "ptlang": "2052",
                "daid": "5",
                "from_ui": "1",
                "pttype": "1",
                "dumy": "",
                "fp": "loginerroralert",
                "action": f"0-0-{int(time.time() * 1000)}",
                "m": "2",
                "g": "1",
                "t": "1",
                "js_type": "0",
                "js_ver": "10291",
                "login_sig": "",
                "pt_randsalt": "0",
            }

            response = await self.client.post(
                url, data=data, cookies=self.cookies
            )
            text = response.text

            if "登录成功" in text or "成功" in text:
                return await self._get_login_token()

            return None

        except Exception as e:
            print(f"Password login request failed: {e}")
            return None

    def is_logged_in(self) -> bool:
        return self.logged_in

    def get_token(self) -> Optional[str]:
        return self.token
