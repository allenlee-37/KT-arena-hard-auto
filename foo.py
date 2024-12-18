from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_response(prompt: str) -> str:
    # 토크나이저 및 모델 초기화
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # eos_token을 pad_token으로 설정
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # 입력 토큰화
    inputs = tokenizer.encode_plus(
        prompt,
        return_tensors="pt",
        padding=True,  # pad_token이 설정되었으므로 가능
        truncation=True,
        max_length=512,
        return_attention_mask=True
    )
    
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # 응답 생성
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,  # 새로운 토큰만 100개 생성
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # 생성된 토큰을 문자열로 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 입력 프롬프트 제거
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

if __name__ == "__main__":
    prompt = "What is a sandwich?"
    response = generate_response(prompt)
    print(response)
