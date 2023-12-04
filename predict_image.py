import torch
from cnn import PyTorchCNN
from skimage import io
import torchvision.transforms as transform

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Initialize the model
pytorch_cnn = PyTorchCNN(num_classes=2)

# Load the model state dict
checkpoint = torch.load('model1.pth')
pytorch_cnn.load_state_dict(checkpoint['model_state_dict'])

# Move the model to the correct device
pytorch_cnn.to(device)


# Assuming X_test is your test data
# Make sure X_test is a torch.Tensor with the appropriate shape
path = 'predict/test3.png'
image = io.imread(path)

image_size = (256, 256)  # Adjust this to the desired size
data_transform = transform.Compose([
    transform.ToPILImage(),  # Convert numpy array to PIL Image
    transform.Grayscale(num_output_channels=3),
    transform.Resize(image_size),
    transform.ToTensor(),
])
X_test = data_transform(image)
X_test = X_test.unsqueeze(0)  # Add a batch dimension
X_test = X_test.to(device)

# Adjust this according to your actual test data
with torch.no_grad():
    predictions = pytorch_cnn.predict(X_test)

# If you have a classification task, you might want to get the predicted class indices
predicted_classes = predictions.item()
pred_label = ['Real life', 'Art']
print(f"Predicted Class: {pred_label[predicted_classes]}")
