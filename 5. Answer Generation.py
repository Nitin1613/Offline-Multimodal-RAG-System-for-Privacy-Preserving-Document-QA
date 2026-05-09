def generate_answer(
    query,
    context_chunks
):

    context_text = "\n\n---\n\n".join(
        context_chunks
    )

    messages = [

        {
            "role": "system",

            "content":
            "Answer ONLY using the provided context."
        },

        {
            "role": "user",

            "content":
            f"""
Context:
{context_text}

Question:
{query}
"""
        }
    ]

    prompt = (
        llm_pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    )

    outputs = llm_pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=True
    )

    generated_text = outputs[0]["generated_text"]

    answer = generated_text[
        len(prompt):
    ].strip()

    return answer
