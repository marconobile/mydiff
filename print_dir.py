import os

def print_dir_structure(start_path, indent=""):
    for item in sorted(os.listdir(start_path)):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            print(f"{indent}📁 {item}/")
            print_dir_structure(path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

# Use current directory or specify your project root
print_dir_structure(".")