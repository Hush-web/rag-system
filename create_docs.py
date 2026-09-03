import os

os.makedirs("docs", exist_ok=True)

for i in range(1, 51):
    with open(f"docs/doc_{i}.txt", "w") as f:
        f.write(f"Document {i}: This is some text about AI and machine learning.\n")
        f.write(f"This is document number {i} in our test dataset.\n")
        f.write(f"It contains information about artificial intelligence.\n")

print("Created 50 test documents in docs/ folder")