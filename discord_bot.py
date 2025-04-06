import os
import discord
from discord.ext import commands
from openai import OpenAI
from dotenv import load_dotenv
from collections import deque
from yusiyeon import YUSIYEON_PROMPT
from kimjiyu import CHARACTER_PROMPT, CHARACTER_NAME

import random
from collections import defaultdict, deque

# 유저별 대화 횟수 카운터
user_message_counts = defaultdict(int)

import asyncio
import functools


# 누적 토큰 수 추적
total_token_usage = 0
# GPT-4o 최신 요금 기준 (2025년 4월)
COST_PER_INPUT_1K = 0.0025  # $2.50 per 1M
COST_PER_OUTPUT_1K = 0.01   # $10.00 per 1M

# 사이트 유도 멘트들
SITE_URL = "https://character-chat.vercel.app/"
PROMO_MESSAGES = [
    f"이 대화 계속하고 싶으면, [RIPLY 해볼래?]({SITE_URL})",
    f"나만의 캐릭터로 나를 표현해보면 더 재밌을 걸? 👉 [RIPLY에서 시작하기]({SITE_URL})",
    f"우리가 나눈 얘기들, 잊지 않게 모아둘 수 있어! 💾 [여기서 확인해봐]({SITE_URL})"
]

# 대화 수 카운터
message_count = 0

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 사용자 ID별로 deque(최근 대화 10개)를 자동 생성
conversation_histories = defaultdict(lambda: deque(maxlen=10))

# 프롬프트 생성 (대화 히스토리 포함)
def generate_prompt(conversation_memory, new_message):
    history = "\n".join(conversation_memory)
    prompt = f"""
{CHARACTER_PROMPT}

아래는 지금까지 나눈 대화입니다:
{history}

사용자의 새 메시지:
{new_message}

캐릭터의 응답:
"""
    return prompt.strip()

# OpenAI 호출 함수
def query_gpt(prompt):
    global total_token_usage
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.8
        )
        # 토큰 사용량
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        total_token_usage += total_tokens

        input_cost = (input_tokens / 1000) * COST_PER_INPUT_1K
        output_cost = (output_tokens / 1000) * COST_PER_OUTPUT_1K
        total_cost = input_cost + output_cost

        print(f"""📊 GPT 응답 요약:
        - 입력 토큰: {input_tokens}
        - 출력 토큰: {output_tokens}
        - 총 토큰: {total_tokens}
        - 예상 비용: ${total_cost:.5f}
        - 누적 토큰: {total_token_usage} tokens
        """)



        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"🚨 OpenAI API 호출 오류: {e}")
        return "답변 생성 중 문제가 생겼어요. 다시 시도해주세요."
        

# Discord 봇 설정
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user.name}이(가) 성공적으로 로그인했습니다!')
    await bot.change_presence(activity=discord.Game(name=f"{CHARACTER_NAME}과 대화 중"))
    # 서버들에서 닉네임 변경
    for guild in bot.guilds:
        try:
            me = guild.get_member(bot.user.id)
            if me:
                await me.edit(nick=CHARACTER_NAME)
                print(f"✅ '{guild.name}' 서버에서 닉네임을 '{CHARACTER_NAME}'으로 변경")
        except Exception as e:
            print(f"❌ '{guild.name}' 서버 닉네임 변경 실패: {e}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message):
        content = message.content.replace(f'<@{bot.user.id}>', '').strip()
        if not content:
            return

        async with message.channel.typing():
            # 유저 고유 ID를 기준으로 대화 히스토리 분리
            user_id = message.author.id
            user_history = conversation_histories[user_id]  # 해당 유저의 deque 가져오기
            # 대화 히스토리에 사용자 메시지 추가
            user_history.append(f"사용자: {content}")
            user_name = message.author.display_name


            # 프롬프트 생성 (해당 유저의 히스토리 기반)
            prompt = generate_prompt(user_history, content)


            start_time = asyncio.get_event_loop().time()

            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                functools.partial(query_gpt, prompt)
            )
            answer = answer.replace("{{user}}", user_name)
            # 🔹 시간 측정 끝
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            print(f"⚡ GPT 응답 시간: {duration:.2f}초")

            # 캐릭터 응답을 히스토리에 추가
            user_history.append(f"캐릭터: {answer}")
            

            # 사용자 대화 수 카운트
            global message_count
            message_count += 1

            # ✅ 5회 이상부터만 유도 멘트 가능
            if message_count >= 5 and random.random() < 0.3:
                promo = random.choice(PROMO_MESSAGES)
                answer += f"\n\n💡 {promo}"
            # Discord 메시지 길이 제한 처리 (2000자)
            if len(answer) > 2000:
                for i in range(0, len(answer), 2000):
                    await message.channel.send(answer[i:i+2000])
            else:
                await message.channel.send(answer)

# 봇 실행 (메인 실행 진입점)
if __name__ == "__main__":
    if DISCORD_BOT_TOKEN:
        bot.run(DISCORD_BOT_TOKEN)
    else:
        print("오류: DISCORD_BOT_TOKEN 환경 변수가 설정되지 않았습니다.")