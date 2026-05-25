from chat.conversation import chat

questions = [
    "What television series is this episode from?",
    "Name of the person who directed it?",
    "Why did she leave the show?"
]

for question in questions:
    print(f"\nQ: {question}")
    answer = chat(question)
    print(f"A: {answer}")