import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transform
loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Load image
def load_image(path):
    image = Image.open(path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device)

content_img = load_image("content.jpg")
style_img = load_image("style.jpg")

# Display image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")

# Load VGG19 model (updated method)
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# Content Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

# Gram Matrix
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# Style Loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

# Layers used
content_layers = ['conv_4']
style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']

content_losses = []
style_losses = []

model = nn.Sequential()

i = 0

for layer in cnn.children():

    if isinstance(layer, nn.Conv2d):
        i += 1
        name = f'conv_{i}'

    elif isinstance(layer, nn.ReLU):
        name = f'relu_{i}'
        layer = nn.ReLU(inplace=False)

    elif isinstance(layer, nn.MaxPool2d):
        name = f'pool_{i}'

    else:
        name = f'layer_{i}'

    model.add_module(name, layer)

    if name in content_layers:
        target = model(content_img).detach()
        content_loss = ContentLoss(target)
        model.add_module(f"content_loss_{i}", content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        target_feature = model(style_img).detach()
        style_loss = StyleLoss(target_feature)
        model.add_module(f"style_loss_{i}", style_loss)
        style_losses.append(style_loss)

# Input image
input_img = content_img.clone()

optimizer = optim.LBFGS([input_img.requires_grad_()])

num_steps = 50
style_weight = 1000000
content_weight = 1

run = [0]

while run[0] <= num_steps:

    def closure():

        input_img.data.clamp_(0,1)

        optimizer.zero_grad()

        model(input_img)

        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss

        for cl in content_losses:
            content_score += cl.loss

        loss = style_weight * style_score + content_weight * content_score

        loss.backward()

        run[0] += 1

        if run[0] % 50 == 0:
            print(f"Step {run[0]}:")
            print(f"Style Loss: {style_score.item():4f} Content Loss: {content_score.item():4f}")

        return loss

    optimizer.step(closure)

# Clamp final image
input_img.data.clamp_(0,1)

# Show result
plt.figure()
imshow(input_img, "Styled Image")
plt.show()

# Save output as JPG
output = input_img.clone().detach().cpu().squeeze(0)
result = transforms.ToPILImage()(output)
result.save("styled_output.jpg", "JPEG")

print("Stylized image saved as styled_output.jpg")