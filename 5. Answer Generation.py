# 3. Prompt Augmentation & Local LLM Summarization
def generate_answer(query, context_chunks):
    """Combines query and context, then queries the local Qwen LLM."""

    context_text = "\n\n---\n\n".join(context_chunks)

    # Format messages using the standard Chat format
    messages =[
        {
            "role": "system",
            "content": "You are a highly capable, local AI assistant. Please answer the user's query based ONLY on the provided context. If the context does not contain the answer, say 'I cannot find the answer in the provided context.'"
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nUser Query:\n{query}"
        }
    ]

    # Apply the model's specific chat template
    prompt = llm_pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate the output
    outputs = llm_pipeline(
        prompt,
        max_new_tokens=256,
        temperature=0.3, # Low temperature for factual RAG responses
        do_sample=True
    )

    # Extract only the generated answer (strip the prompt)
    generated_text = outputs[0]["generated_text"]
    answer = generated_text[len(prompt):].strip()

    return answer