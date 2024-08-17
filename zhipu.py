from transformers import pipeline

# 加载 ChatGLM 模型
chat_glm = pipeline('text-generation', model='THUDM/chatglm-6b', device=0)


# 基于一段文本生成角色人设的函数
def generate_character_profile(text):
    prompt = f"Based on the following text, create a detailed character profile:\n\n{text}"
    result = chat_glm(prompt, max_length=150, do_sample=True, temperature=0.7)
    return result[0]['generated_text'].strip()


# 角色对话生成函数
def generate_dialogue(character_1, character_2, num_turns=5):
    dialogue = []
    for i in range(num_turns):
        speaker = character_1 if i % 2 == 0 else character_2
        prompt = f"Character profile:\n{speaker}\n\nGenerate a reply:"
        reply = chat_glm(prompt, max_length=100, do_sample=True, temperature=0.7)
        dialogue.append(f"Character {i % 2 + 1}: " + reply[0]['generated_text'].strip())
    return dialogue


# 保存对话数据到文件
def save_dialogue_to_file(dialogue, filename='dialogue.txt'):
    with open(filename, 'w') as f:
        for line in dialogue:
            f.write(line + '\n')


# 主函数
if __name__ == "__main__":
    # 示例文本
    text = """
    In a faraway kingdom, there lived a wise and noble king who ruled his land with kindness and justice.
    His only daughter, the princess, was known for her beauty and bravery.
    """

    # 生成角色人设
    character_1 = generate_character_profile(text)
    character_2 = generate_character_profile(text)

    print("Character 1 Profile:\n", character_1)
    print("Character 2 Profile:\n", character_2)

    # 生成对话
    dialogue = generate_dialogue(character_1, character_2, num_turns=5)

    # 打印并保存对话数据
    print("\nGenerated Dialogue:")
    for line in dialogue:
        print(line)

    save_dialogue_to_file(dialogue)
    print(f"\nDialogue saved to 'dialogue.txt'")
