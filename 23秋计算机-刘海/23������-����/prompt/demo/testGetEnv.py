import os
from urllib import response

print(os.getenv("OPENAI_API_KEY"))
print(os.getenv("OPENAI_BASE_URL"))

print(response['choices'][0].message.content)