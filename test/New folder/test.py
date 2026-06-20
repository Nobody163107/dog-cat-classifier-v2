from backend.predictor import predict

ver, pred = predict("test_images/Cat.jpg")

print(ver, pred)