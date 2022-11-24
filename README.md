# StoryPot
GCSWGraduation work

구어체 문장 -> 키워드만 추출 (광고, 숏폼) 데이터셋 1번

두명

(GPT)

광고: 웹툰 표지로 일러스트를 쓰고싶은데 그림 그려줘. 웹툰은
좀비 던전과 보물에 얽힌 비밀들에 관한 내용이고, 내가 필요
한 일러스트는 좀비 던전을 지나는 칼을 든 금발의 5살 여자
아이야.

드라마: 지나가는 자동차에 부딪혀 슬펐다.
자동차, 부딪혔다, 슬펐다.


정답 라벨:  일러스트레이션, 웹툰, 좀비 던전, 칼을 든, 금발, 5살,여자 아이

키워드가 뽑히는지 2번


후드티를 입고 고양이 귀를 한 분홍머리 여자,
 16살  일본 애니메이션 스타일

후드티, 고양이 귀, 분홍머리, 여자, 16살, 일본 애니메이션 스타일

국왕이자 ‘블랙 팬서’인 '티찰라'의 죽음 이후 수많은 강대국으로부터 위협을 받게 된 '와칸다' 동생이 슬펐다


요약 파트

파파고 api를 쓸건지 
구글 번역 api

단어를 생성 ->

류상연, 이서빈

===================================================================
이미지 생성 파트 

1명: https://huggingface.co/spaces/stabilityai/stable-diffusion,   https://github.com/CompVis/stable-diffusion,         |||||||||||||||||||          1명https://github.com/jina-ai/dalle-flow
앞에,뒤에꺼랑


8컷, 4컷

# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
   "CompVis/stable-diffusion-v1-4", 
   use_auth_token=True
).to("cuda")

prompt = input("일러스트레이션, 웹툰, 좀비 던전, 칼을 든, 금발, 5살,여자 아이")
with autocast("cuda"):
    image = pipe(prompt)["sample"][0]  
    
image.save("astronaut_rides_horse.png")

김용겸: 이미지-> stable-diffusion
차원우: 이미지-> dalle-flow


플랫폼 : 웹
장고:1순위

Flask:2순위

===================================================================

플랫폼 : 웹

GAN -> 사용하는거 (가능하면 하는느낌?)


===================================================================

생성요약 AIHUB데이터셋으로 테스트 -> 요야 잘되늕 pko/pko-t5로 테스트 예정

===================================================================
