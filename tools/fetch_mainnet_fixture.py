"""Fetch the real mainnet block used by tests/test_sandwich_detector.py.

Usage:
    python tools/fetch_mainnet_fixture.py tests/vectors/mainnet_block_12775690_traces.json.gz

Block 12775690 (2021-07-06) holds a sandwich on the ShibaSwap BONE/WETH pair
0xefb47fcfcad4f96c83d4ca676842fb03ef20a477: front-run tx 2, four victims
(txs 3-6), back-run tx 7, attacker contract
0x000000000027d2efc283613d0c3e24a8b430c4d8.

The file is the call-trace export that flashbots/mev-inspect-py keeps in its
own test suite (Parity ``trace_block`` traces plus receipts). It is stored
gzip-compressed and byte-for-byte unchanged otherwise; the SHA-256 below is of
the uncompressed JSON. Nothing in it is produced by this project.
"""
import gzip
import hashlib
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/flashbots/mev-inspect-py/main/tests/blocks/12775690.json"
SHA256 = "440a6c54d857fa93be5f88ce2bf98b3df1299c1b09ee23058ec1b5a2f015b000"


def main(out_path: str) -> None:
    with urllib.request.urlopen(URL, timeout=60) as resp:  # noqa: S310 - fixed URL
        raw = resp.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"unexpected content: sha256 {digest}")
    with gzip.GzipFile(out_path, "wb", mtime=0) as fh:
        fh.write(raw)
    print(f"wrote {out_path} ({len(raw)} bytes uncompressed)")


if __name__ == "__main__":
    main(sys.argv[1])
