from torchvision import transforms
from PIL import Image
import torch
from Siamese_model import SiameseNetwork
from ultralytics import YOLO
import torch.nn as nn

class Recognition(nn.Module):
    def __init__(self, semiesz_model,yolo_model,ear_data):
        super(Recognition, self).__init__()

        self.ear_data = ear_data[0]
        self.ear_labels = ear_data[1]
        self.embedding = semiesz_model.embedding
        self.classifier = semiesz_model.classifier
        self.yolo_model = yolo_model
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: Image.fromarray(x)),
            transforms.Resize((100, 100)),  # Resize images 
            # transforms.Grayscale(num_output_channels=3),  # Convert images to grayscale
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])

    def get_ears(self,results,image):
        ears_all = []
        boxes_all = []
        for r in results:
            if len(r.boxes) == 0:
                continue
            # Convert the boxes and scores to tensors
            boxes = torch.tensor([list(map(int, box.xyxy[0])) for box in r.boxes]).float()
            # Plot the remaining boxes
            for box in boxes:
                # Create a figure and axes
                x1, y1, x2, y2 = map(int, box)
                ear = image[y1:y2, x1:x2]
                ears_all.append(ear)
                boxes_all.append((x1, y1, x2, y2))
        return ears_all,boxes_all
    
    def get_distance(self, embeddings1, embeddings2):
        v=torch.abs(embeddings1 - embeddings2)
        v = v.to(self.classifier.weight.dtype)
        return self.classifier(v)
    

    def get_image_vector(self,images):
        images = [self.transform(image) for image in images]
        tensor_images = torch.stack(images)  # Stack images into a single tensor
        with torch.no_grad():
            output = self.embedding(tensor_images)
        return output

    def identify_ear(self,ears_vector):
        label_predictions = []
        for ear_v in ears_vector:
            ear_v_expanded = ear_v.repeat(self.ear_data.shape[0], 1)        
            distances = self.get_distance(ear_v_expanded,self.ear_data)
    
            # Calculate the highest avrage distance for each label
            # Calculate the sum and count of distances for each label
            label_stats = {}
            for label, distance in zip(self.ear_labels, distances):
                if label not in label_stats:
                    label_stats[label] = {'sum': 0, 'count': 0}
                label_stats[label]['sum'] += distance.item()
                label_stats[label]['count'] += 1

            # Calculate the average distance for each label
            avg_distances = {label: stats['sum'] / stats['count'] for label, stats in label_stats.items()}

            # Find the label with the highest average distance
            max_label = max(avg_distances, key=avg_distances.get)
            # If the highest average distance is less than 0.5, append -1 to predictions
            # Otherwise, append the label with the highest average distance
            if abs(avg_distances[max_label]) < 0.5:
                label_predictions.append((-1,avg_distances[max_label]))
            else:
                label_predictions.append((max_label,avg_distances[max_label]))

        return label_predictions

    def forward(self, image,stream=False):
        
        results = self.yolo_model(image,stream=stream)
        
        #if no ears are detected return results
        if not results:  # If `results` is empty
            return results, ([],[])
        # Get the ears from the results
        ears,boxes = self.get_ears(results,image)
        if not ears:
            return results, ([],[])
        ears_vector= self.get_image_vector(ears)
        label_predictions = self.identify_ear(ears_vector)
        
        return results, (boxes,label_predictions)