%% MATLABによる"ディープラーニングで打音検査"
%
% 打音データの取得の取得から、前処理、ネットワークの構築と学習、
% 結果の可視化までのワークフローをご紹介いたします。
%
%
% 参考例題
% https://jp.mathworks.com/help/deeplearning/examples/deep-learning-speech-recognition.html?lang=en
%
%【デモで使用しているToolbox】
%  Signal Processing Toolbox：信号の前処理（ピーク検出）
%  DSP System Toolbox       ：Audio Toolboxに必要
%  Audio Toolbox            ：音声データの録音、オーディオ用のデータストア
%  Deep Learning Toolbox    ：ディープニューラルネットワークの構築と学習
% （オプション）Parallel Computing Toolbox：GPUによる学習・推論の高速化


%% ピーク検出を使ってサンプルデータを取得
fs = 8820;              % データ量削減のため低周波でサンプリング
segmentDuration = 0.25; % 判定に使う信号長[s]
pkh  = 0.05;             % 検出する最少のピークの高さ(maxは1);

%% 正常箇所の打音データを取得（正常部位を叩いてください）
% ウィンドウを閉じると終了です
savedir = 'livedata/normal';
samplePeaks(savedir,fs,segmentDuration,pkh);


%% 異常箇所の打音データを取得（異常部位を叩いてください）
% ウィンドウを閉じると終了です
savedir = 'livedata/abnormal';
samplePeaks(savedir,fs,segmentDuration,pkh);


%% 背景音（10秒放置で約40個のデータを取得します）
% ウィンドウを閉じると終了です
savedir = 'livedata/background';
sampleBG(savedir,fs,segmentDuration)


%% audioDatastoreによる取得済みデータセットの読み込み
datasetdir = 'livedata';
ads = audioDatastore(datasetdir,'IncludeSubfolders',true,'LabelSource','foldernames');

% データの中身を確認
countEachLabel(ads)

%% 学習用、検証用、テスト用にデータセットを分割
[adsTrain,adsValid,adsTest] = splitEachLabel(ads,0.7,0.1);

% データの中身を確認
countEachLabel(adsTrain)

%% 各々のデータセットからスペクトログラムを取得
segmentDuration = 0.25;   % スピーチの期間[s]
frameDuration = 0.01; % スペクトログラム計算を行うデータの長さ[s]
hopDuration = 0.002;   % スペクトログラムのタイムステップ
numBands = 40;         % log-barkフィルタ数（各スペクトログラムの高さ）
epsil = 1e-6;

% スペクトログラムの取得
STrain = speechSpectrograms(adsTrain,segmentDuration,frameDuration,hopDuration,numBands);
STrain = log10(STrain + epsil);
SValid = speechSpectrograms(adsValid,segmentDuration,frameDuration,hopDuration,numBands);
SValid = log10(SValid + epsil);
STest  = speechSpectrograms(adsTest,segmentDuration,frameDuration,hopDuration,numBands);
STest  = log10(STest + epsil);

% ラベルデータの取得
YTrain = adsTrain.Labels;
YValid = adsValid.Labels;
YTest  = adsTest.Labels;

%% 各クラスのデータを一つずつ読み込み確認
reset(adsTrain);

% 学習用データセットから各クラスの一番目のインデックスを取得
idx  = [find(adsTrain.Labels == 'normal',1), ...
        find(adsTrain.Labels == 'abnormal',1) ...
        find(adsTrain.Labels == 'background',1)];

% スペクトログラムと音声による確認
specMin = min(STrain(:));
specMax = max(STrain(:));
for i = 1:numel(idx)
    
    % 波形の表示
    subplot(2,3,i)
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    t  = (0:size(x)-1)/fs;
    plot(t,x); xlim([0,0.25]); ylim([-1 1]);
    grid on; xlabel('振幅'); ylabel('Time[s]'); 
    title(adsTrain.Labels(idx(i)))
    
    % スペクトログラムの表示
    subplot(2,3,i+3)
    pcolor(STrain(:,:,idx(i)))
    caxis([specMin+2 specMax])
    shading flat
    
    % 音の再生
    sound(x)
    
    pause(1)
end


%% データの水増し
sz = size(STrain);
specSize = sz(1:2);
imageSize = [specSize 1];
augmenter = imageDataAugmenter( ...
    'RandXTranslation',[-10 10], ...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,STrain,YTrain, ...
    'DataAugmentation',augmenter);

%% ネットワークの定義
% クラスによってデータ数が異なる（正常なデータの方が多い)
% このため、クラスに応じて学習に重みを付けるカスタムの分類層を使用
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

%% 学習オプションの設定
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

%% 学習の開始
trainedNet = trainNetwork(augimdsTrain,layers,options);

%% ネットワークの評価(混同行列による可視化)
YPred= classify(trainedNet,STest);
figure('Units','normalized');
cm = confusionchart(YTest,YPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%% Liveストリーミングによる検証

% Webカメラを同時に使用する場合
% - useCamera=trueに設定してください
% - OS上でWebカメラを使用するように設定してください
% - webcamコマンドでWebカメラを選択してください
%   選択可能なカメラのリストはwebcamlistコマンドで確認できます
useCamera = false;


% 各種設定値の定義
%fs = 8820; % サンプリング周波数
%segmentDuration = 0.25; % スピーチの期間[s]
samplesPerFrame = ceil(fs*segmentDuration);
frameDuration = 0.01;   % スペクトログラム計算を行うデータの長さ[s]
hopDuration = 0.002;    % スペクトログラムのタイムステップ
numBands = 40;          % log-barkフィルタ数（各スペクトログラムの高さ）
epsil = 1e-6;

% 音声データ読み込みオブジェクトの生成
audioIn = audioDeviceReader(fs, samplesPerFrame);

% カメラとの接続
if useCamera
    cam = webcam(2);
    hcam = figure('Units','normalized','Position',[0.5 0.0 0.5 1]);
    axcam = axes(hcam);
end

% 判定と可視化
h = figure('Units','normalized','Position',[0.0 0.0 0.5 1]);

specMin = -6;     % min(STrain(:));
specMax = 0.6184; % max(STrain(:));
while ishandle(h)
    % 音声データを抽出し、バッファに格納
    x = audioIn();
    % スペクトログラムの計算
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
        
    % スペクトログラム画像の幅が短いときのためにパディング処理
    X = zeros([numBands,numHops],'single');
    w = size(spec,2);
    left = floor((numHops-w)/2)+1;
    ind = left:left+w-1;
    X(:,ind) = spec;
    X = log10(X + epsil);
    
    % スペクトログラムを使って分類
    [YPredicted,probs] = classify(trainedNet,X);
    
    % 音声波形の表示
    subplot(2,1,1);
    t = (0:size(x,1)-1)/fs;   
    plot(t,x)
    ylim([-1 1]); grid on; xlabel('Time[s]');
    
    % スペクトログラムの表示
    subplot(2,1,2)
    pcolor(X)
    caxis([specMin+2 specMax])
    shading flat
    
    % 判定結果の可視化
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

% 各オブジェクトの解放
release(audioIn);
if useCamera
    clear cam
end


%% Copyright 2019 The MathWorks, Inc.