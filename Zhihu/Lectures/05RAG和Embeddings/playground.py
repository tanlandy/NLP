from os.path import dirname, join
current_dir = dirname(__file__)
file_path = join(current_dir, "./whatto.md")


with open(file_path, 'r', encoding='utf-8') as f:
    markdown_content = f.read()

# Step 2: Print or process the content
print(markdown_content)