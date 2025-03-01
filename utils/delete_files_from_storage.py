from openai import OpenAI
import datetime

# Initialize the OpenAI client
with open("../tokens/openai_token.txt") as f:
    openai_token = f.read()
client = OpenAI(
    # This is the default and can be omitted
    api_key=openai_token,
)


def upload_file():
    filename = input("Enter the filename to upload: ")
    try:
        with open(filename, "rb") as file:
            response = client.files.create(file=file, purpose="assistants")
            print(response)
            print(f"File uploaded successfully: {response.filename} [{response.id}]")
    except FileNotFoundError:
        print("File not found. Please make sure the filename and path are correct.")


def list_files():
    response = client.files.list(purpose="assistants")
    if len(response.data) == 0:
        print("No files found.")
        return
    for file in response.data:
        created_date = datetime.datetime.utcfromtimestamp(file.created_at).strftime(
            "%Y-%m-%d"
        )
        print(f"{file.filename} [{file.id}], Created: {created_date}")


def n_list_files():
    response = client.files.list(purpose="assistants")
    if len(response.data) == 0:
        print("No files found.")
        return
    print(f"count = {len(response.data)}")


def list_and_delete_file():
    while True:
        response = client.files.list(purpose="assistants")
        files = list(response.data)
        if len(files) == 0:
            print("No files found.")
            return
        for i, file in enumerate(files, start=1):
            created_date = datetime.datetime.utcfromtimestamp(file.created_at).strftime(
                "%Y-%m-%d"
            )
            print(f"[{i}] {file.filename} [{file.id}], Created: {created_date}")
        choice = input(
            "Enter a file number to delete, or any other input to return to menu: "
        )
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(files):
            return
        selected_file = files[int(choice) - 1]
        client.files.delete(selected_file.id)
        print(f"File deleted: {selected_file.filename}")


def delete_all_files():
    confirmation = input(
        "This will delete all OpenAI files with purpose 'assistants'.\n Type 'YES' to confirm: "
    )
    if confirmation == "YES":
        response = client.files.list(purpose="assistants")
        for i, file in enumerate(response.data):
            client.files.delete(file.id)
            if i % 10 == 0:
                print(i)
        print("All files with purpose 'assistants' have been deleted.")
    else:
        print("Operation cancelled.")


def main():
    while True:
        print("\n== Assistants file utility ==")
        print("[1] Upload file")
        print("[2] List all files")
        print("[3] List all and delete one of your choice")
        print("[4] Delete all assistant files (confirmation required)")
        print("[5] Count of all files")
        print("[9] Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            upload_file()
        elif choice == "2":
            list_files()
        elif choice == "3":
            list_and_delete_file()
        elif choice == "4":
            delete_all_files()
        elif choice == "5":
            n_list_files()
        elif choice == "9":
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
