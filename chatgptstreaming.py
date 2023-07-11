import openai
from tokenize import tokenize
from io import BytesIO
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()

openai.api_key = ""

# Context size of the conversion, excluding the system message
CONTEXT_SIZE = 8
MAX_TOKEN = 4050
RESPONSE_SIZE = 1000
SAFEGUARD_SIZE = 30
CONVERSATIONS = []


class Conversation:
    def __init__(self, role, content, token_count):
        self.role = role 
        self.content = content
        self.token_count = token_count

    def to_gpt_message(self):
        return {"role": self.role, "content": self.content}


def get_token_count(message):
    return len(list(tokenize(BytesIO(message.encode('utf-8')).readline)))


def getContextSize():
    global CONVERSATIONS
    count = 0
    for c in CONVERSATIONS:
        count += c.token_count

    return count


def appendToConversation(message, f):
    global CONVERSATIONS, CONTEXT_SIZE
    CONVERSATIONS.append(message)
    if len(CONVERSATIONS) > CONTEXT_SIZE + 1:
        print(f"{Fore.RED}The oldest context is removed due to the size limit of the context{Style.RESET_ALL}")
        CONVERSATIONS.pop(1)

    f.write(f"{message.role}:{message.content}\n")
    f.flush()


def generate_completions(model, response_size):
    print(f"GPT is thinking...")

    global CONVERSATIONS, RESPONSE_SIZE
    response = openai.ChatCompletion.create(
        messages=[c.to_gpt_message() for c in CONVERSATIONS],
        max_tokens=response_size,
        model=model,
        stream=True
    )

    output = ""
    completion_token_count = 0
    for message in response:
        completion_token_count = completion_token_count + 1
        if message["choices"] is not None and message["choices"][0] is not None and message["choices"][0]["delta"] is not None:
            if "content" in message["choices"][0]["delta"]:
                current = message["choices"][0]["delta"]["content"]
                output += current
                print(f"{Fore.GREEN}{current}{Style.RESET_ALL}", end='')
            else:
                output += '\n\t'
                print('')

    print(f"{Fore.YELLOW}Context size: {len(CONVERSATIONS - 1)}{Style.RESET_ALL}")
    return (output, completion_token_count)

def run(model):
    f = open("/Users/haomingyuan/Documents/scripts/gpt_dump.txt", "a")

    try:
        f.write(f"===================================================================\n")

        system_message = "You are a helpful assistant."

        # It's tested manually that the system message above is 12 tokens. DO NOT CHANGE
        appendToConversation(Conversation("system", system_message, 12), f)

        print(f"{Fore.GREEN}How can I help today{Style.RESET_ALL}")

        global CONVERSATIONS, SAFEGUARD_SIZE

        while True:
            prompt = input(">> ")

            if prompt == "new":
                CONVERSATIONS = [CONVERSATIONS[0]]
                print(f"{Fore.GREEN}Alright, starting a new conversion now...{Style.RESET_ALL}")
                continue
            elif prompt == "quit":
                print(f"{Fore.GREEN}Exiting the chat...{Style.RESET_ALL} ")
                break

            appendToConversation(Conversation("user", prompt, get_token_count(prompt) + SAFEGUARD_SIZE), f)

            # Drop the context of the response size is smaller than 100
            responseSize = MAX_TOKEN - getContextSize()

            while responseSize < RESPONSE_SIZE:
                CONVERSATIONS.pop(1)
                responseSize = MAX_TOKEN - getContextSize()

            (output, completion_token_count) = generate_completions(model, responseSize)

            appendToConversation(Conversation("assistant", output, completion_token_count), f)

            print(f"{Fore.RED}Token left:{MAX_TOKEN - getContextSize()}{Style.RESET_ALL}")


    finally:
        f.close()


if __name__ == "__main__":
    model = "gpt-3.5-turbo"
    try:
        run(model)
    except KeyboardInterrupt:
        # do nothing here
        pass