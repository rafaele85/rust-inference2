import struct

with open("./model/model.safetensors", "rb") as f:
    header_len = struct.unpack("<Q", f.read(8))[0]
    print(f"Header length: {header_len} bytes")
