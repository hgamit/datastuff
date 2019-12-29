%% ABC_M1.m
% ABC_M1('25_1_1', 1)
% ABC_M1('23_3_1', 1)
% ABC_M1('23_3_1_3', 1)
% ABC_M1('23_1_3', 1)
function ABC_M1(DataShape, MyBatchSize)
disp('ABC_LSTM.m ============================================')

outputMode = 'last';

AlphabetChar = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
AlphabetIntNoNorm = [1 : 26]';
AlphabetIntNorm = [1 : 26]' / 26;

tmp = str2double(split(DataShape, '_'))';
NumSeqs = tmp(1);       SeqLen = tmp(2);        NumVars = tmp(3); 
if strcmp(DataShape, '25_1_1')
    %% 25_1_1
    NumVars = 1;
    AlphabetInt_YTrain = categorical(AlphabetIntNoNorm(2 : end));
    AlphabetInt = AlphabetIntNorm(1 : end - 1);
elseif strcmp(DataShape, '23_3_1') | strcmp(DataShape, '23_3_1_3')
    %% 23_3_1
    if length(tmp) > 3
        outputMode = 'sequence';
        OutputLen = tmp(2);     % output seq_len = input seq_len for this study here
    end
    NumVars = 1;
    AlphabetInt = [];       Y_tmp = {};
    for i = 1 : 23
        AlphabetInt = [AlphabetInt; AlphabetIntNorm(i : i+SeqLen-1)'];
        if length(tmp) == 3             % M:1
            Y_tmp = [Y_tmp; categorical(AlphabetIntNoNorm(i+SeqLen))];   % get next char
        elseif length(tmp) > 3          % M:M
            Y_tmp = [Y_tmp; {categorical(AlphabetIntNoNorm(i+1 : i+SeqLen)')}]; 
        end
    end
    AlphabetInt_YTrain = Y_tmp;
elseif strcmp(DataShape, '23_1_3')
    %% 23_1_3
    NumVars = 3;
    AlphabetInt = [];       Y_tmp = [];
    for i = 1 : 23
        AlphabetInt = [AlphabetInt; AlphabetIntNorm(i : i+NumVars-1)'];
        Y_tmp = [Y_tmp; AlphabetIntNoNorm(i+SeqLen)];
    end
    AlphabetInt_YTrain = categorical(Y_tmp);
end

%%
AlphabetInt = reshape(AlphabetInt, [NumVars, SeqLen, NumSeqs]);
AlphabetInt_XTrain = squeeze(mat2cell(AlphabetInt, [NumVars], [SeqLen], ones(NumSeqs, 1)));

%{
[AlphabetInt_XTrain, AlphabetInt_YTrain, AlphabetIntNoNorm, AlphabetInt] = ...
            PrepareSeqData(AlphabetInt, NumVars, SeqLen, NumSeqs);
%}


%%
outputSize = 32;
numClasses = length(unique(AlphabetInt_YTrain));

layers = [ ...
    sequenceInputLayer(NumVars)
    lstmLayer(outputSize, 'OutputMode', outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 500;
miniBatchSize = MyBatchSize;

%{
    'GradientThreshold',1, ...
    'SequenceLength', 'longest', ...             % 'longest'
    'Shuffle','every-epoch', ...
%}
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'auto', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', MyBatchSize, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% Train LSTM Network
net = trainNetwork(AlphabetInt_XTrain, AlphabetInt_YTrain, layers, options);

%% classify all training input
YPred = classify(net, AlphabetInt_XTrain, 'MiniBatchSize', miniBatchSize);

CFM = confusionmat(AlphabetInt_YTrain, YPred);
Acc = round(sum(diag(CFM))/sum(CFM(:)), 2);
disp(Acc)
disp(YPred')

%% demonstrate a random starting point
net = resetState(net);   
disp('random starting point')
for RandomChar = [2 3 9 10 11  20 21 22]
    YPredOne = classify(net, RandomChar, 'MiniBatchSize', miniBatchSize);
    disp([RandomChar, double(YPredOne)])
end

%{
%% stateful
net = resetState(net);   
disp('Stateful...')
for CharIdx = 1 : length(AlphabetInt_XTrain)
    Char = AlphabetInt_XTrain(CharIdx);
    [net, YPredScores] = predictAndUpdateState(net, Char, 'MiniBatchSize', miniBatchSize);
    [maxV YPredOne] = max(YPredScores);
    disp([Char{:}, YPredOne])
end
%}

disp('============================================================')

i = 0;
