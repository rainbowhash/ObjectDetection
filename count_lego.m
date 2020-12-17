function [numA,numB]=count_lego(I)
%Input:RGB Image,containg legos I
%Output:numA,numB ie the count of red2*2 and blue 2*4 lego.
%The method that I have incorporated is using both image processing and
%machine learning algorithms. First, the input is enhanced to increase
%the contrast, brightness and
%then it is thresholded to remove only blue and red blocks in
%the image. The thresholded image  split using region prop and each
%region is passed to the machine learning algorithm to predict if the 
%2*4 blue or 2*2 red bricks exist.
%Because the model is trained with very limited images, I am considering
%the cluster of predictions in a region to be true i.e. if a region as
%whole has many predictions, I sum that prediction and have a threshold
%, if that summation is similar to the threshold score I assume the cube
%exist there

clc;  % Clear command window.
clear;  % Delete all variables.
close all;  % Close all figure windows except those created by imtool.
imtool close all;  % Close all figure windows created by imtool.
workspace;  % Make sure the workspace panel is showing.

image=imread("training_images/train04.jpg");

%enhance brightness and contrast for both blue and red regions seperatly
Blue_enhanced_image=imadjust(image,[0,0.9],[0 1]);
Red_enhanced_image=imadjust(image,[0.1,0.9],[0 1]);

%threshold to seperate red and blue blocks
Tred=createMaskRed(Red_enhanced_image);
Tblue=createMaskBlue(Blue_enhanced_image);
%figure(1),imshow(image);

%Loading pretrained model for both red and blue blocks
persistent s detector 
if isempty(detector)
  	s = coder.load('detectorblue.mat');
	detector = acfObjectDetector(s.Classifier,s.TrainingOptions);
end
persistent y rdetector
if isempty(rdetector)
  	y = coder.load('detectorRed.mat');
	rdetector = acfObjectDetector(y.Classifier,y.TrainingOptions);
end


%blue blocks prediction
%masking out all region except blue
inv=~Tblue;
Blue_enhanced_image(inv)=255;
%identifying the blocks
bstats = regionprops(Tblue,'all');
blue_count=0;
figure(2),imshow(image)

for i =1:length(bstats)
    BB = bstats(i).BoundingBox;
    if bstats(i).Area>30000 && bstats(i).Area<60000 
    subImage = imcrop(Blue_enhanced_image,BB);
    [bboxes, scores] = detect(detector,subImage);
     if length(scores)>2
        rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;
        blue_count=blue_count+1;
        %display(length(scores));
    end
    end
end  

%Red blocks prediction 

%masking out all region except red
rred=~Tred;
Red_enhanced_image(rred)=255;
%figure(3),imshow(image);
bstats = regionprops(Tred,'all');
red_count=0;
figure(4),imshow(image)
for i =1:length(bstats)
    BB = bstats(i).BoundingBox;
    if bstats(i).Area>10000 && bstats(i).Area<60000
    subImage = imcrop(Red_enhanced_image,BB);
    [bboxes, scores] = detect(rdetector,subImage);
     if length(scores)<7
        rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',2) ;
        red_count=red_count+1;
        %display(length(scores))
     end
    end
end
numA=blue;
numB=red;
end

