%% サンプルデータの取得
function sampleBG(savedir,fs,frameLength)

samplesPerFrame = ceil(fs*frameLength);
audioIn = audioDeviceReader(fs, samplesPerFrame);

% 保存フォルダの確認（データがある場合は追加していく仕様） 
dirinfo = dir(fullfile(savedir,'*.wav'));
if isempty(dirinfo)
    count = 1;
    mkdir(savedir)
else
    count = numel({dirinfo.name}) + 1;
end

h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
while ishandle(h)
   x = audioIn();
   x = x(:,1);
   t = (0:size(x,1)-1)/fs;
   
   % ライブデータの表示
   subplot(1,2,1)
   plot(t,x)
   ylim([-1 1]); grid on; xlabel('Time[s]');
   title('Live Data','FontSize',18);
   
   
   locs = 1;
   
   if ~isempty(locs)
       
       % オーディオファイルの保存
       filename = sprintf('%04i.wav',count);
       audiowrite(fullfile(savedir,filename),x,fs);
       count = count + 1;

       % 検出したデータの表示
       subplot(1,2,2)
       plot(t,x)
       ylim([-1 1]); grid on; xlabel('Time[s]');
       title({'Detected signal',filename},'FontSize',18);

   end
   
   drawnow 

end
release(audioIn)

end
%% Copyright 2019 The MathWorks, Inc.