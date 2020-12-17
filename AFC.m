%Script that trains the ACF object detector
%load labeled images
new = load('training_images/red-res-train.mat');
trainingDataTable = objectDetectorTrainingData(new.gTruth);

%start training
acfDetector = trainACFObjectDetector(trainingDataTable,'NegativeSamplesFactor',3,"NumStages",8);

%save the model to a file
s = toStruct(acfDetector);
save('detectorRed.mat','-struct','s') 

%test the model prediction
img = imread('stopSignTest.jpg');
img = imread("training_images/train08.jpg");
[bboxes,scores] = detect(acfDetector,img);
for i = 1:length(scores)
   annotation = sprintf('Confidence = %.1f',scores(i));
   img = insertObjectAnnotation(img,'rectangle',bboxes(i,:),annotation);
end
figure
imshow(img)