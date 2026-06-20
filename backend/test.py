from shap_utils import generate_shap, processor

filer, files = processor("/mnt/c/Users/Praneeth Tadi/Documents/Coding/Machine Learning/ML Projects/Dog-cat-classify/dog-cat-classify/dog4.jpg")

file = generate_shap(filer, files)

print(file)