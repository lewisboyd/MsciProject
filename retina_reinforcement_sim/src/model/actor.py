class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(95040, 512)
        self.head = nn.Linear(512, 3)

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(1, 95040)
        x = F.relu(self.fc1(x))
        x = self.head(x)
        return x
