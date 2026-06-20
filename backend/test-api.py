import requests
import os

correct = 0
total = 0

for label in ["cats", "dogs"]:

    folder = f"/mnt/c/Users/Praneeth Tadi/Documents/Coding/Machine Learning/ML Projects/Dog-cat-classify/dog-cat-classify/catsvsdogs/test/{label}"

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        with open(path, "rb") as img:

            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": img}
            )

        try:
            data = response.json()
            prediction = data["Verdict"]
        except Exception as e:
            print("\nFAILED FILE:", path)
            print("STATUS:", response.status_code)
            print("RESPONSE:", response.text)
            break

        if prediction == label[:-1]:
            correct += 1

        total += 1
        # if(total == 40): 
        #     break

print(f"Accuracy = {correct}/{total}")
print(f"Accuracy = {100*correct/total:.2f}%")