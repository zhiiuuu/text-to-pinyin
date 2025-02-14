import re
from typing import Union, List

from fastapi import FastAPI, Body
from paddlespeech.t2s.frontend import TextNormalizer
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from pydantic import BaseModel, Field

from app.vo import BaseResponse
from config import API_PREFIX


class TextRequestModel(BaseModel):
    sentence: Union[str, List[str]] = Field(None, description='sentence(s)')


class TextResponseModel(BaseModel):
    sentences: List[str] = Field(None, description='input sentences')
    result: List[List[str]] = Field(None, description='output sentences')


def mount_routes(app: FastAPI, args):
    api_prefix = f'{API_PREFIX}/text2pinyin'
    api_tag = 'Text2Pinyin'

    @app.post(f'{api_prefix}/text-normalize', tags=[api_tag], summary='List')
    async def text_normalize(
            body: TextRequestModel = Body(..., examples=[{
                "sentence": [
                    "电影中梁朝伟扮演的陈永仁的编号27149",
                    "这块黄金重达324.75克",
                    "我们班的最高总分为583分",
                    "12~23、-1.5~2",
                    "她出生于86年8月18日，她弟弟出生于1995年3月1日",
                    "等会请在12:05请通知我",
                    "今天的最低气温达到-10°C",
                    "现场有7/12的观众投出了赞成票",
                    "明天有62％的概率降雨",
                    "随便来几个价格12块5，34.5元，20.1万",
                    "这是固话0421-33441122",
                    "这是手机+86 18544139121"
                ],
            }])
    ) -> BaseResponse:
        sentences = [body.sentence] if isinstance(body.sentence, str) else body.sentence
        sentences = [x.strip() for x in sentences if x.strip()]

        if len(sentences) == 0:
            return BaseResponse(data=TextResponseModel(sentences=[], result=[]))

        normalizer = TextNormalizer()
        result = [
            normalizer.normalize(x) for x in sentences
        ]

        return BaseResponse(data=TextResponseModel(sentences=sentences, result=result))

    @app.post(f'{api_prefix}/text-to-pinyin', tags=[api_tag], summary='Get')
    async def text_to_pinyin(
            body: TextRequestModel = Body(..., examples=[{
                'sentence': [
                    '电影中梁朝伟扮演的陈永仁的编号二七一四九', '这块黄金重达三百二十四点七五克',
                ],
            }])
    ) -> BaseResponse:
        # 获取请求中的句子列表，如果是单个字符串则转换为列表
        sentences = [body.sentence] if isinstance(body.sentence, str) else body.sentence
        # 去除句子列表中空字符串
        sentences = [x.strip() for x in sentences if x.strip()]

        # 如果没有有效句子，返回空结果
        if len(sentences) == 0:
            return BaseResponse(data=TextResponseModel(sentences=[], result=[]))

        frontend = Frontend()
        result = []
        for sentence in sentences:
            # 去除非中文字符
            chinese_sentence = extract_chinese(sentence)
            if not chinese_sentence.strip():
                result.append([])
                continue

            try:
                # 获取拼音结果
                pinyin_result = frontend.g2pW_model(chinese_sentence)
                if pinyin_result:
                    result.append(pinyin_result[0])
                else:
                    result.append([])
            except Exception as e:
                # 处理转换过程中可能出现的异常
                print(f"Error converting sentence {sentence}: {e}")
                result.append([])

        # 返回包含原始句子和拼音结果的响应
        return BaseResponse(data=TextResponseModel(sentences=sentences, result=result))

    # 获取字符串中文部分
    def extract_chinese(text: str) -> str:
        # 匹配所有中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        # 将匹配到的中文字符列表拼接成字符串
        return ''.join(chinese_chars)

    pass
