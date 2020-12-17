new = load('training_images/red-res-train.mat');
trainingDataTable = objectDetectorTrainingData(new.gTruth);
acfDetector = trainACFObjectDetector(trainingDataTable,'NegativeSamplesFactor',3,"NumStages",8);
s = toStruct(acfDetector);
save('detectorRed.mat','-struct','s') 
img = imread('stopSignTest.jpg');

img=imread("training_images/train08.jpg");
[bboxes,scores] = detect(acfDetector,img);
for i = 1:length(scores)
   annotation = sprintf('Confidence = %.1f',scores(i));
   img = insertObjectAnnotation(img,'rectangle',bboxes(i,:),annotation);
   
end

figure
imshow(img)