import ollama

response = ollama.list()

# == Chat example ==
res = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {"role": "user", "content": "why is the sky blue?"},
    ],
)
# print("# == Chat example == \n")
# print(res["message"]["content"])
# print("\n")

# == Chat example streaming ==
res = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "why is the ocean so salty?",
        },
    ],
    stream=True,
)

# print("# == Chat example streaming == \n")
# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True)
# print("\n")

# == Generate example ==
res = ollama.generate(
    model="llama3.2:3b",
    prompt="why is the sky blue?",
)
# print("# == Generate example == \n")
# print(res["response"])
# print("\n")

# print(ollama.show("llama3.2:3b"))

# Create a new model with modelfile
print("# == Create a new model with modelfile == \n")

#modelfile = """
#FROM llama3.2:3b
#SYSTEM You are very smart assistant who knows everything about oceans. You are very succinct and informative.
#PARAMETER temperature 0.1
#"""

# Fix: Pass the modelfile parameter
ollama.create(model='knowitall', from_='llama3.2:3b', system="You are very smart assistant who knows everything about oceans. You are very succinct and informative.")

res = ollama.generate(model="knowitall", prompt="why is the ocean so salty?")
print(res["response"])

# delete model
ollama.delete("knowitall")
