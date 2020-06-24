%% MATLAB�ɂ��"�f�B�[�v���[�j���O�őŉ�����"
%
% �ŉ��f�[�^�̎擾�̎擾����A�O�����A�l�b�g���[�N�̍\�z�Ɗw�K�A
% ���ʂ̉����܂ł̃��[�N�t���[�����Љ�����܂��B
%
%
% �Q�l���
% https://jp.mathworks.com/help/deeplearning/examples/deep-learning-speech-recognition.html?lang=en
%
%�y�f���Ŏg�p���Ă���Toolbox�z
%  Signal Processing Toolbox�F�M���̑O�����i�s�[�N���o�j
%  DSP System Toolbox       �FAudio Toolbox�ɕK�v
%  Audio Toolbox            �F�����f�[�^�̘^���A�I�[�f�B�I�p�̃f�[�^�X�g�A
%  Deep Learning Toolbox    �F�f�B�[�v�j���[�����l�b�g���[�N�̍\�z�Ɗw�K
% �i�I�v�V�����jParallel Computing Toolbox�FGPU�ɂ��w�K�E���_�̍�����


%% �s�[�N���o���g���ăT���v���f�[�^���擾
fs = 8820;              % �f�[�^�ʍ팸�̂��ߒ���g�ŃT���v�����O
segmentDuration = 0.25; % ����Ɏg���M����[s]
pkh  = 0.05;             % ���o����ŏ��̃s�[�N�̍���(max��1);

%% ����ӏ��̑ŉ��f�[�^���擾�i���핔�ʂ�@���Ă��������j
% �E�B���h�E�����ƏI���ł�
savedir = 'livedata/normal';
samplePeaks(savedir,fs,segmentDuration,pkh);


%% �ُ�ӏ��̑ŉ��f�[�^���擾�i�ُ핔�ʂ�@���Ă��������j
% �E�B���h�E�����ƏI���ł�
savedir = 'livedata/abnormal';
samplePeaks(savedir,fs,segmentDuration,pkh);


%% �w�i���i10�b���u�Ŗ�40�̃f�[�^���擾���܂��j
% �E�B���h�E�����ƏI���ł�
savedir = 'livedata/background';
sampleBG(savedir,fs,segmentDuration)


%% audioDatastore�ɂ��擾�ς݃f�[�^�Z�b�g�̓ǂݍ���
datasetdir = 'livedata';
ads = audioDatastore(datasetdir,'IncludeSubfolders',true,'LabelSource','foldernames');

% �f�[�^�̒��g���m�F
countEachLabel(ads)

%% �w�K�p�A���ؗp�A�e�X�g�p�Ƀf�[�^�Z�b�g�𕪊�
[adsTrain,adsValid,adsTest] = splitEachLabel(ads,0.7,0.1);

% �f�[�^�̒��g���m�F
countEachLabel(adsTrain)

%% �e�X�̃f�[�^�Z�b�g����X�y�N�g���O�������擾
segmentDuration = 0.25;   % �X�s�[�`�̊���[s]
frameDuration = 0.01; % �X�y�N�g���O�����v�Z���s���f�[�^�̒���[s]
hopDuration = 0.002;   % �X�y�N�g���O�����̃^�C���X�e�b�v
numBands = 40;         % log-bark�t�B���^���i�e�X�y�N�g���O�����̍����j
epsil = 1e-6;

% �X�y�N�g���O�����̎擾
STrain = speechSpectrograms(adsTrain,segmentDuration,frameDuration,hopDuration,numBands);
STrain = log10(STrain + epsil);
SValid = speechSpectrograms(adsValid,segmentDuration,frameDuration,hopDuration,numBands);
SValid = log10(SValid + epsil);
STest  = speechSpectrograms(adsTest,segmentDuration,frameDuration,hopDuration,numBands);
STest  = log10(STest + epsil);

% ���x���f�[�^�̎擾
YTrain = adsTrain.Labels;
YValid = adsValid.Labels;
YTest  = adsTest.Labels;

%% �e�N���X�̃f�[�^������ǂݍ��݊m�F
reset(adsTrain);

% �w�K�p�f�[�^�Z�b�g����e�N���X�̈�Ԗڂ̃C���f�b�N�X���擾
idx  = [find(adsTrain.Labels == 'normal',1), ...
        find(adsTrain.Labels == 'abnormal',1) ...
        find(adsTrain.Labels == 'background',1)];

% �X�y�N�g���O�����Ɖ����ɂ��m�F
specMin = min(STrain(:));
specMax = max(STrain(:));
for i = 1:numel(idx)
    
    % �g�`�̕\��
    subplot(2,3,i)
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    t  = (0:size(x)-1)/fs;
    plot(t,x); xlim([0,0.25]); ylim([-1 1]);
    grid on; xlabel('�U��'); ylabel('Time[s]'); 
    title(adsTrain.Labels(idx(i)))
    
    % �X�y�N�g���O�����̕\��
    subplot(2,3,i+3)
    pcolor(STrain(:,:,idx(i)))
    caxis([specMin+2 specMax])
    shading flat
    
    % ���̍Đ�
    sound(x)
    
    pause(1)
end


%% �f�[�^�̐�����
sz = size(STrain);
specSize = sz(1:2);
imageSize = [specSize 1];
augmenter = imageDataAugmenter( ...
    'RandXTranslation',[-10 10], ...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,STrain,YTrain, ...
    'DataAugmentation',augmenter);

%% �l�b�g���[�N�̒�`
% �N���X�ɂ���ăf�[�^�����قȂ�i����ȃf�[�^�̕�������)
% ���̂��߁A�N���X�ɉ����Ċw�K�ɏd�݂�t����J�X�^���̕��ޑw���g�p
% https://jp.mathworks.com/help/deeplearning/ug/create-custom-weighted-cross-entropy-classification-layer.html

classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));

dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer(imageSize)
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([1 13])
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

%% �w�K�I�v�V�����̐ݒ�
miniBatchSize = 64;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{SValid,YValid}, ...
    'ValidationFrequency',validationFrequency+1, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.01, ...
    'LearnRateDropPeriod',20);

%% �w�K�̊J�n
trainedNet = trainNetwork(augimdsTrain,layers,options);

%% �l�b�g���[�N�̕]��(�����s��ɂ�����)
YPred= classify(trainedNet,STest);
figure('Units','normalized');
cm = confusionchart(YTest,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%% Live�X�g���[�~���O�ɂ�錟��

% Web�J�����𓯎��Ɏg�p����ꍇ
% - useCamera=true�ɐݒ肵�Ă�������
% - OS���Web�J�������g�p����悤�ɐݒ肵�Ă�������
% - webcam�R�}���h��Web�J������I�����Ă�������
%   �I���\�ȃJ�����̃��X�g��webcamlist�R�}���h�Ŋm�F�ł��܂�
useCamera = false;


% �e��ݒ�l�̒�`
%fs = 8820; % �T���v�����O���g��
%segmentDuration = 0.25; % �X�s�[�`�̊���[s]
samplesPerFrame = ceil(fs*segmentDuration);
frameDuration = 0.01;   % �X�y�N�g���O�����v�Z���s���f�[�^�̒���[s]
hopDuration = 0.002;    % �X�y�N�g���O�����̃^�C���X�e�b�v
numBands = 40;          % log-bark�t�B���^���i�e�X�y�N�g���O�����̍����j
epsil = 1e-6;

% �����f�[�^�ǂݍ��݃I�u�W�F�N�g�̐���
audioIn = audioDeviceReader(fs, samplesPerFrame);

% �J�����Ƃ̐ڑ�
if useCamera
    cam = webcam(2);
    hcam = figure('Units','normalized','Position',[0.5 0.0 0.5 1]);
    axcam = axes(hcam);
end

% ����Ɖ���
h = figure('Units','normalized','Position',[0.0 0.0 0.5 1]);

specMin = -6;     % min(STrain(:));
specMax = 0.6184; % max(STrain(:));
while ishandle(h)
    % �����f�[�^�𒊏o���A�o�b�t�@�Ɋi�[
    x = audioIn();
    % �X�y�N�g���O�����̌v�Z
    frameLength = round(frameDuration*fs);
    hopLength = round(hopDuration*fs);
    numHops = ceil((segmentDuration - frameDuration)/hopDuration);
    
    spec = melSpectrogram(x,fs, ...
        'WindowLength',frameLength, ...
        'OverlapLength',frameLength - hopLength, ...
        'FFTLength',1024, ...
        'NumBands',numBands);
        %'FrequencyRange',[50,7000]);
    epsil = 1e-6;
        
    % �X�y�N�g���O�����摜�̕����Z���Ƃ��̂��߂Ƀp�f�B���O����
    X = zeros([numBands,numHops],'single');
    w = size(spec,2);
    left = floor((numHops-w)/2)+1;
    ind = left:left+w-1;
    X(:,ind) = spec;
    X = log10(X + epsil);
    
    % �X�y�N�g���O�������g���ĕ���
    [YPredicted,probs] = classify(trainedNet,X);
    
    % �����g�`�̕\��
    subplot(2,1,1);
    t = (0:size(x,1)-1)/fs;   
    plot(t,x)
    ylim([-1 1]); grid on; xlabel('Time[s]');
    
    % �X�y�N�g���O�����̕\��
    subplot(2,1,2)
    pcolor(X)
    caxis([specMin+2 specMax])
    shading flat
    
    % ���茋�ʂ̉���
    subplot(2,1,1);
    if YPredicted == "background" 
       title(" ")
    else
       title(string(YPredicted),'FontSize',20)
    end
    
    if useCamera
        img = snapshot(cam);
        imshow(img,'Parent',axcam)
    end
    
    drawnow
    
end

% �e�I�u�W�F�N�g�̉��
release(audioIn);
if useCamera
    clear cam
end


%% Copyright 2019 The MathWorks, Inc.