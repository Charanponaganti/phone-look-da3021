import re
import urllib.request

req = urllib.request.Request("https://aiff22.github.io/", headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
links = re.findall(r'href=["\']([^"\']+)["\']', html)
print("ALL HREFS:")
for x in links:
    print(x)
print("\nZIPS:")
for x in re.findall(r"https?://[^\s\"']+\.zip", html):
    print(x)
print("\nSAMPLE CONTEXT:")
for m in re.finditer(r".{0,100}sample.{0,160}", html, re.I):
    print(m.group(0))
    print("---")
# also check github readme
req2 = urllib.request.Request(
    "https://raw.githubusercontent.com/aiff22/DPED/master/README.md",
    headers={"User-Agent": "Mozilla/5.0"},
)
try:
    md = urllib.request.urlopen(req2, timeout=60).read().decode("utf-8", "ignore")
    print("\nGITHUB README LINKS:")
    for x in re.findall(r"https?://[^\s)\]]+", md):
        print(x)
except Exception as e:
    print("readme fail", e)
