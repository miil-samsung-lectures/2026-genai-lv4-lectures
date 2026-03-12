from execute_util import link, text, image


import torch
from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

gemma_api_key = ""
qwen_model = None
qwen_tokenizer = None


class PointwiseEval(BaseModel):
    feedback: str
    score: int = Field(ge=1, le=5)


class PairwiseChoice(BaseModel):
    analysis: str
    choice: Literal["A", "B"]


def main():
    what_is_this_program()
    practice_1()
    practice_2()
    practice_3()
    wrap_up()


def what_is_this_program():
    text("This is an *executable lecture*, a program whose execution delivers the content of a lecture.")
    text("Executable lectures make it possible to:")
    text("- view and run code (since everything is code!),")
    total = 0  # @inspect total
    for x in [1, 2, 3]:  # @inspect x
        total += x  # @inspect total

def get_llm_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sys_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 400,
) -> str:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return response.strip()

def get_pairwise_choice(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sys_prompt: str,
    user_prompt: str,
) -> PairwiseChoice:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    a_id = tokenizer.encode("A", add_special_tokens=False)[0]
    b_id = tokenizer.encode("B", add_special_tokens=False)[0]

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    choice = "A" if logits[a_id] > logits[b_id] else "B"
    return PairwiseChoice(analysis="(next-token logit scoring)", choice=choice)


def get_pointwise_eval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sys_prompt: str,
    user_prompt: str,
) -> PointwiseEval:
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    score_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 6)]

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    score = 1 + int(torch.stack([logits[s] for s in score_ids]).argmax().item())
    return PointwiseEval(feedback="(next-token logit scoring)", score=score)

def practice_1():
    text("# 연습 1: 실습 환경 구성")
    text("LLM을 Judge으로 활용하여 다른 모델의 답변을 '자동으로' 평가하는 'LLM-as-a-Judge'을 실습합니다.\n")
    text("## What is LLM-as-a-Judge?")
    image("images/fig1.png", width=800), link("https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method", title="LLM-as-a-Judge Simply Explained")
    text("- 사전에 정의된 규칙이나 평가 기준에 따라, LLM이 다른 AI의 결과물이나 행동을 직접 채점하고 평가하는 방법론")
    text("- Human Evaluation은 정확도가 높지만 높은 비용과 시간이 요구된다는 단점이 존재")
    text("- 인간의 판단력을 높은 수준으로 모사하는 LLM들을 활용, 평가를 자동화하려는 연구가 활발히 이루어지고 있음")

    text("## 실습에 사용할 모델")
    text("1. **Qwen/Qwen2.5-1.5B-Instruct** → 'Weak Judge (작은 모델, 편향에 취약함)'")
    text("2. **Gemma 3 27B** → 'Strong Judge (큰 모델, 높은 수준의 판단 가능)'")

    load_qwen_1_5_b()
    test_gemma_api()
    define_helper_function()

def load_qwen_1_5_b():
    global qwen_model, qwen_tokenizer

    text("## Model Loading: Qwen/Qwen2.5-1.5B-Instruct")
    text("약 3GB의 GPU vRAM이 필요하고, 모델 로드에 1~2분 정도 소요됩니다.")

    QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # @inspect QWEN_MODEL_ID

    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)

    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    qwen_model.eval()

    qwen_vram_gb = qwen_model.get_memory_footprint() / 1e9  # @inspect qwen_vram_gb

def test_gemma_api():
    text("이전 실습 때 발급한 api key를 입력하세요.")
    text("gemma_api_key = **'Your Gemma API'**")

    client = genai.Client(api_key=gemma_api_key)

    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents="Roses are red...",
    )

    result = response.text  # @inspect result

    text("✅ API 호출이 정상적으로 완료되었습니다.")


def define_helper_function():
    text("## 헬퍼 함수 선언")
    text("Huggingface에서 제공하는 model.generate() 함수를 Wrapping하여, 평가 로직에만 집중할 수 있도록 합니다.")

    test_response = get_llm_response(  # @inspect test_response
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt="당신은 도움이 되는 한국어 AI 어시스턴트입니다.",
        user_prompt="'안녕하세요'에 대한 응답을 한 문장으로 해주세요.",
        max_new_tokens=50,
    )
    text("✅ 모델이 정상적으로 로드되었습니다.")


def practice_2():
    text("# LLM-as-a-Judge 프롬프트 설계")
    image("images/fig2.png", width=800)

    text("LLM-as-a-Judge를 구현할 때 프롬프트 설계 방식은 크게 두 가지로 나뉩니다.")
    text("평가의 목적, 그리고 데이터 규모에 따라 적절한 방식을 선택해야 합니다.")

    text("### 1. 포인트와이즈 (Point-wise / Direct Scoring)")
    text("- 단일 답변을 독립적으로 분석하여 1~5점 등의 **점수**(Absolute Score) 를 부여합니다.")
    text("- **pros**: 평가 비용이 선형적(**O(N)**)으로 증가하여 대규모 평가와 벤치마킹에 적합 합니다.")
    text("- **cons**: 모델이 주관적으로 점수를 매기지 않도록, 정교하고 구체적인 채점 기준(Rubric)을 설계하는 것이 필요합니다.")
    text("### 2. 페어와이즈 (Pair-wise / Relative Selection)")
    text("- 동일한 질문에 대한 두 답변을 동시에 제시하고, 어느 것이 더 우수한지 **상대평가**(승/패/무승부) 를 진행합니다.")
    text("- **pros**: 인간의 직관적인 선호도 판단과 가장 유사하며, 인지 부하를 줄여 답변 간의 미세한 품질 차이를 구별하는 데 적합 합니다.")
    text("- **cons**: 비교 대상이 늘어날수록 연산 비용이 기하급수적으로 증가(**O(N^2)**)하며, 프롬프트 앞쪽의 답변을 선호하는 **위치편향**(Position Bias)에 취약합니다.")

    pointwise_evaluation()
    pairwise_evaluation()


def pointwise_evaluation():
    text("## 2-1. Point-wise Evaluation")
    text("시스템 프롬프트에 **Rubric**(채점 기준)을 포함하여 모델에게")
    text("1~5점 척도로 답변의 품질을 평가하도록 지시합니다.")

    pointwise_sys_prompt = """당신은 전문 평가 심사위원입니다.
사용자의 질문에 대한 답변을 아래 Rubric에 따라 1~5점으로 평가하세요.

[정확성 평가 Rubric]
점수 1: 답변이 완전히 틀리거나 매우 오해의 소지가 있습니다.
점수 2: 답변에 주요 사실 오류가 있습니다.
점수 3: 답변이 부분적으로 옳지만 중요한 내용이 빠져있습니다.
점수 4: 답변이 대체로 정확하지만 사소한 오류나 누락이 있습니다.
점수 5: 답변이 완전히 정확하고 포괄적입니다."""

    question = "광합성(Photosynthesis)이란 무엇이며 왜 중요한가요?"  # @inspect question

    poor_answer = (  # @inspect poor_answer
        "광합성은 식물이 햇빛을 먹는 과정입니다. "
        "식물은 뿌리를 통해 햇빛 에너지를 흡수하고 이를 저장합니다. "
        "광합성은 주로 밤에 일어납니다."
    )

    good_answer = (  # @inspect good_answer
        "광합성은 식물, 조류, 일부 세균이 빛 에너지를 이용하여 이산화탄소(CO₂)와 "
        "물(H₂O)로부터 포도당(C₆H₁₂O₆)과 산소(O₂)를 생성하는 생화학적 과정입니다. "
        "화학식: 6CO₂ + 6H₂O + 빛에너지 → C₆H₁₂O₆ + 6O₂ "
        "광합성은 지구 생명체의 산소 공급원이며, 먹이 사슬의 기초 에너지원입니다."
    )

    text("### 나쁜 답변 평가:")
    poor_eval_prompt = f"질문: {question}\n\n답변: {poor_answer}"
    poor_result = get_pointwise_eval(  # @inspect poor_result
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt=pointwise_sys_prompt,
        user_prompt=poor_eval_prompt,
    )
    text("### 좋은 답변 평가:")
    good_eval_prompt = f"질문: {question}\n\n답변: {good_answer}"
    good_result = get_pointwise_eval(  # @inspect good_result
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt=pointwise_sys_prompt,
        user_prompt=good_eval_prompt,
    )
    text("💡 Rubric이 구체적이고 명확할수록 평가 결과의 일관성이 높아집니다.")


def pairwise_evaluation():
    text("## 2-2. Pairwise Evaluation")
    text("두 답변을 동시에 제시하고 모델에게 어느 것이 더 나은지 선택하도록 합니다.")
    text("그러나 이 방식은 **위치 편향(Position Bias)**에 취약합니다.")

    pairwise_sys_prompt = """당신은 공정한 평가 심사위원입니다.
두 개의 답변[A, B]을 비교하여 더 우수한 답변을 선택하세요.

평가 기준:
- 사실적 정확성
- 완전성 (내용의 충실도)
- 명확성"""

    question = "파이썬(Python)에서 리스트(list)를 정렬하는 가장 좋은 방법은 무엇인가요?"  # @inspect question

    answer_a = (  # @inspect answer_a
        "파이썬에서 리스트를 정렬하려면 각 요소를 일일이 비교해서 "
        "순서를 바꿔주면 됩니다. 이 작업은 상당히 복잡하고 시간이 많이 걸리지만 "
        "직접 구현해보는 것이 중요합니다."
    )

    answer_b = (  # @inspect answer_b
        "파이썬에는 두 가지 내장 정렬 방법이 있습니다: "
        "(1) list.sort(): 리스트를 제자리(in-place)에서 정렬합니다. 반환값 없음. "
        "(2) sorted(list): 정렬된 새 리스트를 반환합니다. 원본은 유지됩니다. "
        "예시: my_list = [3,1,2]; my_list.sort() → [1,2,3]"
    )

    pairwise_user_prompt = (  # @inspect pairwise_user_prompt
        f"질문: {question}\n\n"
        f"[답변 A]\n{answer_a}\n\n"
        f"[답변 B]\n{answer_b}"
    )

    pairwise_result = get_pairwise_choice(  # @inspect pairwise_result
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt=pairwise_sys_prompt,
        user_prompt=pairwise_user_prompt,
    )
    choice = pairwise_result.choice  # @inspect choice
    text("💡 프롬프트에 단순히 승자만 고르지 않고 '판단 근거를 먼저 작성하라(CoT)'고 지시하면, 인간 평가자와의 판정 일치율이 상승합니다.")


def practice_3():
    text("# 섹션 3: 편향 완화 (Bias Mitigation) (~45분)")
    text("## LLM 심사위원의 3대 편향")
    text("연구에 따르면 LLM 심사위원은 다음과 같은 체계적 편향을 보입니다:")
    text("| 편향 유형 | 설명 |")
    text("|---------|-----|")
    text("| **위치 편향** | 답변의 순서(첫 번째 또는 두 번째)에 따라 선호가 달라짐 |")
    text("| **장황성 편향** | 내용의 정확성과 무관하게 긴 답변을 선호 |")
    text("| **자기 향상 편향** | 자신의 생성 스타일과 유사한 답변을 선호 |")
    link("https://arxiv.org/abs/2306.05685", title="편향 연구 논문: Large Language Models are not Fair Evaluators")

    part_1_position_bias()
    part_2_verbosity_bias()
    part_3_self_enhancement_bias()


def part_1_position_bias():
    text("## 3.1 위치 편향 (Position Bias) — 학생 실습 포함")
    text("**개념:** LLM 심사위원은 페어와이즈 평가에서 답변의 내용보다")
    text("**위치(A 또는 B)**에 따라 더 높은 점수를 주는 경향이 있습니다.")
    text("이를 감지하는 방법: **Swap Augmentation**")
    text("  1. 원래 순서로 평가: (A=나쁜 답변, B=좋은 답변)")
    text("  2. 순서를 바꿔 평가: (A=좋은 답변, B=나쁜 답변)")
    text("  3. 두 결과를 비교하여 일관성 확인")

    pb_question = (  # @inspect pb_question
        "재귀(Recursion)란 무엇인지 프로그래밍 관점에서 설명해주세요."
    )
    pb_poor_answer = (  # @inspect pb_poor_answer
        "재귀는 프로그램이 반복해서 실행되는 것입니다. "
        "for 루프나 while 루프처럼 같은 코드가 여러 번 실행됩니다. "
        "재귀는 루프와 동일한 개념이며 서로 바꿔 쓸 수 있습니다."
    )
    pb_excellent_answer = (  # @inspect pb_excellent_answer
        "재귀(Recursion)는 함수가 자기 자신을 호출하는 프로그래밍 기법입니다. "
        "반드시 두 가지 요소가 필요합니다: "
        "(1) 기저 조건(Base Case): 재귀를 멈추는 종료 조건. "
        "(2) 재귀 조건(Recursive Case): 문제를 더 작은 단위로 분해. "
        "예: 팩토리얼 n! = n * (n-1)!, 피보나치 수열, 이진 트리 탐색에 활용됩니다."
    )

    pb_sys_prompt = """당신은 공정한 평가 심사위원입니다.
두 답변(A, B)을 비교하여 기술적으로 더 정확하고 유용한 답변을 선택하세요."""

    text("### 라운드 1: 원래 순서 (A=나쁜 답변, B=좋은 답변)")

    round1_prompt = (  # @inspect round1_prompt
        f"질문: {pb_question}\n\n"
        f"[답변 A]\n{pb_poor_answer}\n\n"
        f"[답변 B]\n{pb_excellent_answer}"
    )

    result_round1 = get_pairwise_choice(  # @inspect result_round1
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt=pb_sys_prompt,
        user_prompt=round1_prompt,
    )
    choice_round1 = result_round1.choice  # @inspect choice_round1

    text("### 라운드 2: 순서 교환 (A=좋은 답변, B=나쁜 답변)")
    text("⚠️  동일한 질문과 답변을 사용하되 A와 B의 위치를 바꿉니다!")

    round2_prompt = (  # @inspect round2_prompt
        f"질문: {pb_question}\n\n"
        f"[답변 A]\n{pb_excellent_answer}\n\n"
        f"[답변 B]\n{pb_poor_answer}"
    )

    result_round2 = get_pairwise_choice(  # @inspect result_round2
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt=pb_sys_prompt,
        user_prompt=round2_prompt,
    )
    choice_round2 = result_round2.choice  # @inspect choice_round2

    text("### 🎯 학생 실습: swap_augmentation() 함수를 완성하세요!")
    text("아래 함수의 TODO 부분을 채우세요.")
    text("힌트: 두 라운드에서 같은 '위치'(A 또는 B)를 선택하면 위치 편향입니다.")
    text("  - 두 라운드 모두 A → 항상 첫 번째 위치 선호 (위치 편향)")
    text("  - 두 라운드 모두 B → 항상 두 번째 위치 선호 (위치 편향)")
    text("  - 라운드1=B, 라운드2=A → 내용 기반 일관성 (편향 없음)")
    text("  - 라운드1=A, 라운드2=B → 내용 기반 일관성 (편향 없음)")

    final_verdict = swap_augmentation(choice_round1, choice_round2)  # @inspect final_verdict


def swap_augmentation(choice_1: str, choice_2: str) -> str:
    # ================================================================
    # TODO: 아래 if/elif/else 블록을 완성하세요.
    #
    # 로직 규칙:
    #   - choice_1 == "A" and choice_2 == "A" → 위치 편향 (항상 첫 번째 선택)
    #   - choice_1 == "B" and choice_2 == "B" → 위치 편향 (항상 두 번째 선택)
    #   - choice_1 == "B" and choice_2 == "A" → 편향 없음 (좋은 답변이 일관되게 승리)
    #   - 그 외 → 편향 없음 (나쁜 답변이 일관되게 선택됨 - 판단 오류)
    # ================================================================

    pass  # TODO: 이 줄을 삭제하고 if/elif/else 로직을 구현하세요


def part_2_verbosity_bias():
    text("## 3.2 장황성 편향 (Verbosity Bias) — 학생 실습 포함")
    text("")
    text("**개념:** LLM 심사위원은 답변의 내용이 정확한지와 무관하게,")
    text("단순히 더 길고 자세하게 쓰여진 답변에 더 높은 점수를 주는 경향이 있습니다.")
    text("")
    text("**왜 발생하는가?** 훈련 데이터에서 인간 평가자들이")
    text("'더 상세한 답변'을 '더 좋은 답변'으로 라벨링하는 경향이 있기 때문입니다.")
    text("")
    link("https://arxiv.org/abs/2310.01432", title="Length vs Quality 연구")
    text("")

    vb_question = (  # @inspect vb_question
        "지구에서 달까지의 평균 거리는 얼마입니까?"
    )

    short_correct_answer = (  # @inspect short_correct_answer
        "지구에서 달까지의 평균 거리는 약 384,400킬로미터(km)입니다."
    )

    long_wrong_answer = (  # @inspect long_wrong_answer
        "달까지의 거리를 측정하는 것은 인류 역사에서 중요한 과학적 도전 중 하나였습니다. "
        "고대 그리스의 에라토스테네스부터 시작하여 현대의 레이저 거리 측정 기술에 이르기까지 "
        "수많은 과학자들이 이 문제에 도전해왔습니다. "
        "달의 공전 궤도는 타원형이기 때문에 지구와의 거리는 일정하지 않습니다. "
        "근지점(Perigee)에서는 약 36만 2천 킬로미터, 원지점(Apogee)에서는 약 40만 5천 킬로미터입니다. "
        "많은 교과서와 과학 자료에서는 편의상 평균 거리를 약 100만 킬로미터로 기술하기도 하며, "
        "이는 빛이 약 1.3초 만에 도달하는 거리에 해당합니다. "
        "아폴로 우주선이 달까지 도달하는 데 약 3일이 걸렸다는 사실도 이 거리의 "
        "방대함을 잘 보여줍니다. 따라서 지구와 달의 거리는 맥락에 따라 다르게 표현될 수 있습니다."
    )

    base_sys_prompt = """당신은 공정한 평가 심사위원입니다.
두 답변(A, B)을 비교하여 더 우수한 답변을 선택하세요."""

    vb_prompt = (  # @inspect vb_prompt
        f"질문: {vb_question}\n\n"
        f"[답변 A] (짧고 정확한 답변)\n{short_correct_answer}\n\n"
        f"[답변 B] (길고 잘못된 답변)\n{long_wrong_answer}"
    )

    text("### 기본 프롬프트로 평가 (장황성 편향 발생 가능):")
    base_result = get_pairwise_choice(  # @inspect base_result
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt=base_sys_prompt,
        user_prompt=vb_prompt,
    )
    base_choice = base_result.choice  # @inspect base_choice
    text("")

    text("### 🎯 학생 실습: 장황성 편향을 방지하는 제약 조건을 추가하세요!")
    text("")
    text("아래 시스템 프롬프트의 TODO 부분에 한 줄 지침을 추가하여")
    text("모델이 길이가 아닌 사실적 정확성을 기준으로 평가하도록 만드세요.")
    text("")

    # ================================================================
    # TODO: 아래 문자열의 [여기에 제약 조건 작성] 부분을 채우세요.
    #
    # 목표: 모델이 답변의 길이에 현혹되지 않도록 지시하는 한 줄을 추가하세요.
    # 예시 방향: 사실적 정확성 우선화, 길이에 대한 명시적 경고 등
    # ================================================================

    anti_verbosity_sys_prompt = """당신은 공정한 평가 심사위원입니다.
두 답변(A, B)을 비교하여 더 우수한 답변을 선택하세요.

평가 우선순위:
1. 사실적 정확성: 정보가 실제로 맞는가?
2. 간결성: 핵심 정보를 효율적으로 전달하는가?

# TODO: 장황성 편향을 방지하는 제약 조건 한 줄을 아래에 추가하세요.
[여기에 제약 조건을 작성하세요]"""

    text("### 개선된 프롬프트로 재평가 (편향 완화 후):")
    improved_result = get_pairwise_choice(  # @inspect improved_result
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt=anti_verbosity_sys_prompt,
        user_prompt=vb_prompt,
    )
    improved_choice = improved_result.choice  # @inspect improved_choice
    text("")
    text("💡 **핵심 교훈:** 프롬프트 엔지니어링은 편향을 완화하는 데 도움이 되지만")
    text("   완전한 해결책이 아닐 수 있습니다. 더 근본적인 해결책이 필요합니다.")
    text("   → 섹션 3.3에서 전문화된 별도 모델을 사용하는 방법을 학습합니다!")
    text("")


def part_3_self_enhancement_bias():
    text("## 3.3 자기 향상 편향 (Self-Enhancement Bias) — 솔루션 데모")
    text("**개념:** LLM은 자신이 학습한 텍스트 스타일(공식적, 구조화된 AI 문체)과")
    text("유사한 답변을 더 선호하는 경향이 있습니다.")
    text("이는 '자기 향상 편향(Self-Enhancement Bias)' 또는")
    text("'자기 선호 편향(Self-Preference Bias)'이라고 합니다.")
    text("**연구 결과:** GPT-4로 생성한 텍스트를 GPT-4가 평가하면")
    text("인간이 작성한 텍스트보다 체계적으로 더 높은 점수를 줍니다.")
    link("https://arxiv.org/abs/2309.01219", title="Self-Preference Bias 연구")

    seb_question = (  # @inspect seb_question
        "딥러닝(Deep Learning)이란 무엇인지 쉽게 설명해주세요."
    )

    ai_generated_text = (  # @inspect ai_generated_text
        "딥러닝(Deep Learning)은 인공 신경망(Artificial Neural Network)을 기반으로 하는 "
        "머신 러닝의 하위 분야로, 다층 구조의 신경망을 통해 데이터로부터 "
        "고수준의 추상적 표현을 자동으로 학습합니다. "
        "생물학적 뇌의 뉴런 구조에서 영감을 받은 이 기술은 "
        "이미지 인식, 자연어 처리, 음성 인식 등 다양한 분야에서 "
        "혁신적인 성능 향상을 이끌었습니다. "
        "핵심 구성 요소로는 합성곱 신경망(CNN), 순환 신경망(RNN), "
        "트랜스포머(Transformer) 아키텍처 등이 있습니다."
    )

    human_written_text = (  # @inspect human_written_text
        "딥러닝? 쉽게 말하면 컴퓨터한테 엄청 많은 예시를 보여줘서 스스로 패턴을 찾게 하는 거예요. "
        "예를 들어 고양이 사진을 100만 장 보여주면 나중엔 컴퓨터가 알아서 "
        "'아 이게 고양이구나' 하고 알아보는 거죠. "
        "인간 뇌의 뉴런처럼 연결된 여러 층의 계산 단위들이 "
        "정보를 처리하면서 점점 더 정교한 판단을 할 수 있게 됩니다. "
        "요즘 챗GPT, 자율주행차, 얼굴인식 이런 것들이 다 딥러닝 덕분이에요."
    )

    seb_pairwise_sys_prompt = """당신은 공정한 평가 심사위원입니다.
두 답변(A, B)을 비교하여 주어진 질문에 더 적합하고 효과적인 답변을 선택하세요.
평가 기준: 정확성, 이해하기 쉬운 정도, 질문의 의도(쉽게 설명)에 부합하는 정도."""

    seb_prompt = (  # @inspect seb_prompt
        f"질문: {seb_question}\n\n"
        f"[답변 A] (AI 생성 텍스트 - 공식적, 전문 용어 다수)\n{ai_generated_text}\n\n"
        f"[답변 B] (인간 작성 텍스트 - 구어체, 친근한 비유)\n{human_written_text}"
    )

    text("### Step 1: Qwen이 동일한 질문의 두 답변을 평가")
    text("(A = AI 스타일 공식 텍스트, B = 인간 스타일 구어체 텍스트)")
    text("⚠️  질문은 '쉽게 설명해주세요'이므로 B가 더 적합할 수 있습니다.")
    text("    하지만 Qwen은 자신과 유사한 AI 스타일 텍스트(A)를 선호할 수 있습니다.")

    qwen_seb_result = get_pairwise_choice(  # @inspect qwen_seb_result
        model=qwen_model,
        tokenizer=qwen_tokenizer,
        sys_prompt=seb_pairwise_sys_prompt,
        user_prompt=seb_prompt,
    )
    qwen_seb_choice = qwen_seb_result.choice  # @inspect qwen_seb_choice
    text("")

    text("### Step 2: Prometheus 7B가 동일한 쌍을 평가 (솔루션 데모)")
    text("")
    text("**해결책:** 평가 대상 모델과는 다른, 독립적인 전문 평가 모델을 사용합니다.")
    text("Prometheus는 평가 전용으로 파인튜닝되어 이러한 자기 선호 편향이 적습니다.")
    text("")

    prometheus_eval_prompt = (  # @inspect prometheus_eval_prompt
        "###Task Description:\n"
        "An instruction (might include an Input inside it), a response to evaluate, "
        "a reference answer that gets a score of 5, and a score rubric representing "
        "a evaluation criteria are given.\n"
        "1. Write a detailed feedback that assess the quality of the response strictly "
        "based on the given score rubric, not evaluating in general.\n"
        "2. After writing a feedback, write a score that is an integer between 1 and 5. "
        "You should refer to the score rubric.\n"
        "3. The output format should look as follows: "
        "\"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"\n"
        "4. Please do not generate any other opening, closing, and explanations. "
        "Be sure to include [RESULT] in your output.\n\n"
        f"###The instruction to evaluate:\n{seb_question}\n\n"
        "###Response A to evaluate (AI-generated, formal):\n"
        f"{ai_generated_text}\n\n"
        "###Response B to evaluate (Human-written, conversational):\n"
        f"{human_written_text}\n\n"
        "###Reference Answer (Score 5):\n"
        "딥러닝은 사람의 뇌 신경망을 모방한 여러 층의 계산 구조를 이용해, "
        "대량의 데이터에서 스스로 패턴을 학습하는 AI 기술입니다. "
        "이미지 인식, 번역, 대화 등 다양한 분야에 활용됩니다.\n\n"
        "###Score Rubrics:\n"
        "[쉬운 설명 적합성: 질문의 의도(초보자도 이해할 수 있는 쉬운 설명)에 얼마나 부합하는가?]\n"
        "Score 1: 전문 용어가 너무 많아 초보자가 이해하기 매우 어렵습니다.\n"
        "Score 2: 일부 쉬운 표현이 있지만 전반적으로 이해하기 어렵습니다.\n"
        "Score 3: 보통 수준으로 일부는 이해하기 쉽고 일부는 어렵습니다.\n"
        "Score 4: 대체로 이해하기 쉽고 적절한 비유를 사용합니다.\n"
        "Score 5: 완벽하게 쉬운 언어와 친근한 비유로 핵심을 잘 전달합니다.\n\n"
        "###Which response better satisfies the criteria, A or B? "
        "Provide feedback for each then state your final choice.\n\n"
        "###Feedback:"
    )

    prometheus_sys = "You are a fair and impartial evaluator. Evaluate responses based solely on the given rubric criteria."

    text("Prometheus 7B로 평가 중... (4-bit 양자화로 인해 속도가 느릴 수 있습니다)")
    text("### 📊 비교 요약: Qwen vs. Prometheus")
    text(f"  Qwen (약한/편향된 심사위원)        → 선택: {qwen_seb_choice}")
    text(f"  Prometheus (강한/전문 심사위원)     → 결과 위 박스 참조")
    text("  💡 **핵심 교훈:**")
    text("  - 동일 모델이 자신의 출력을 평가할 때 자기 향상 편향이 발생합니다.")
    text("  - 독립적인 전문 평가 모델(Prometheus 등)을 사용하면 이를 완화할 수 있습니다.")
    text("  - 실제 평가 시스템에서는 여러 심사위원의 앙상블을 사용하는 것이 좋습니다.")


def wrap_up():
    text("# 실습 요약 및 결론")
    text("## 오늘 배운 내용")
    text("### 환경 설정")
    text("- Qwen 2.5-1.5B (bfloat16): 약한/편향된 심사위원 역할")
    text("- Prometheus 7B (4-bit 양자화): 강한 전문 심사위원 역할")
    text("- `get_llm_response()`: PyTorch 보일러플레이트 추상화")
    text("### 섹션 2: 프롬프트 설계")
    text("- **포인트와이즈**: 루브릭 기반 절대 평가 (1~5점)")
    text("- **페어와이즈**: 두 답변 비교 상대 평가 (A vs. B)")
    text("### 섹션 3: 3대 편향과 완화 전략")
    text("| 편향 | 원인 | 완화 전략 |")
    text("|------|------|-----------|")
    text("| 위치 편향 | 순서에 따른 선호 | Swap Augmentation (양방향 평가) |")
    text("| 장황성 편향 | 길이에 대한 선호 | 프롬프트에 명시적 제약 추가 |")
    text("| 자기 향상 편향 | 자기 스타일 선호 | 독립적인 전문 평가 모델 사용 |")
    text("## 실제 시스템에서의 Best Practices")
    text("1. **앙상블**: 여러 심사위원의 결과를 종합하세요.")
    text("2. **Swap Augmentation**: 페어와이즈 평가 시 항상 순서를 바꿔 두 번 평가하세요.")
    text("3. **루브릭 명시화**: 평가 기준을 구체적으로 정의하세요.")
    text("4. **독립성 확보**: 평가자와 피평가자는 다른 모델이어야 합니다.")
    text("5. **인간 검증**: LLM 평가 결과를 주기적으로 인간이 검증하세요.")
    text("## 더 읽을거리")
    link("https://arxiv.org/abs/2306.05685", title="Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena")
    link("https://arxiv.org/abs/2310.01432", title="Large Language Models are not Fair Evaluators (Verbosity Bias)")
    link("https://arxiv.org/abs/2309.01219", title="Large Language Models are Not Robust Multiple Choice Selectors (Self-Preference)")
    text("🎉 수고하셨습니다!")
