import orjson
from fastapi.encoders import jsonable_encoder
from fastapi_cache import Coder
from typing import List, Any


class ORJsonCoder(Coder):
    @classmethod
    def encode(cls, value: Any) -> bytes:
        print('\n'*100,value)
        return orjson.dumps(
            value,
            default=jsonable_encoder,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )

    @classmethod
    def decode(cls, value: bytes) -> Any:
        return orjson.loads(value)
