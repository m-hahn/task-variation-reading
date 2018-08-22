neatQA = {}
assert(false)

neatQA.number_of_LSTM_layers = 1

neatQA.ALSO_DO_LANGUAGE_MODELING = true

neatQA.ACCESS_MEMORY = false

neatQA.INITIALIZE_FROM_NEAT = true --false--true
if (not neatQA.INITIALIZE_FROM_NEAT) then
 print("NOT INITIALIZING FROM NEAT -- WARNING!!!")
end

neatQA.RUN_THE_FULL_MODULE = true


neatQA.maximalLengthOccurringInInput = {0}

require('nn.RecursorMod')
require('nn.SequencerMod')
require('nn.PrintLayer')
require('nn.BlockGradientLayer')
require('nn.ConstantLayer')
require('nn.VerifyDimensionsLayer')
require('nn.DynamicallySelectTable')

--neatQA.DO_LENGTH_ADAPTIVE = false

-- input: recurrent state and memory cells
-- output: softmax layer
function neatQA.createSimpleAnswerNetwork()
  local model = nn.Sequential()
  model:add(nn.JoinTable(2))
  model:add(nn.Linear(2*params.rnn_size,NUMBER_OF_ANSWER_OPTIONS))
  model:add(nn.LogSoftMax())
--print(model:getParameters():norm())
--crash()
  return model
end


function neatQA.buildNeuralGraph()
  if neatQA.ACCESS_MEMORY then
    crash()
  end

  local inputWords = nn.Identity()()
  local readerModule = autoencoding.create_network(neatQA.ALSO_DO_LANGUAGE_MODELING, true, true):cuda()

--  feedbackModule.add(nn.Identity())
  --feedbackModule.add(nn.SelectTable(2))

--  local transferModule = nn.Identity()
  local transferModule = readerModule

  --local mergeModule = nn.ParallelTable()
--  mergeModule.add(nn.SelectTable(1))

local reader2Rec
if false then

  local startModule = nn.Sequential()
  --startModule:add(nn.VerifyDimensionsLayer({torch.LongTensor({60})}))
  startModule:add(nn.SelectTable(1))
  --startModule:add(nn.VerifyDimensionsLayer(torch.LongTensor({60})))

  local startModuleConcat = nn.ConcatTable()
--  startModule:add(nn.PrintLayer("start 5136"))
  startModuleConcat:add(nn.Identity())
  startModuleConcat:add(nn.ConstantLayer(torch.zeros(params.batch_size,params.rnn_size):cuda()))
  startModuleConcat:add(nn.ConstantLayer(torch.zeros(params.batch_size,params.rnn_size):cuda()))

  startModule:add(startModuleConcat)
--  startModule:add(nn.PrintLayer("start 5139"))

  local inputModule = nn.Identity()

  local feedbackModule = nn.Identity()

--   local wiringExtInput = nn.VerifyDimensionsLayer(torch.Tensor({60,512}))(nn.Identity()())
   local wiringExtInput = nn.Identity()()
   local wiringExtInputSelect = nn.SelectTable(1)(wiringExtInput)
   local wiringRecInput = nn.Identity()()
--   local wiringRecInputPost = nn.PrintLayer("wiringRec")(nn.SelectTable(1)(wiringRecInput))
   local wiringRecFirst = nn.SelectTable(1)(wiringRecInput)
   local wiringRecSecond = nn.SelectTable(2)(wiringRecInput)
  local mergeModule = nn.gModule({wiringExtInput,wiringRecInput},{wiringExtInputSelect,wiringRecFirst,wiringRecSecond})




  reader2Rec = nn.Recurrent(startModule, inputModule, feedbackModule, transferModule, 999, mergeModule)
else
  reader2Rec = nn.RecursorMod(readerModule, nil, {reader_c[0], reader_h[0]}, 1, 3, {true,true,false}):cuda()
end

local withActor
local lastStates
local reader2

  reader2 = nn.Sequencer(reader2Rec)(inputWords)
  reader2 = nn.Identity()({reader2, nn.DynamicallySelectTable(neatQA.maximalLengthOccurringInInput)(reader2) } )

  lastStates = nn.SelectTable(2)(reader2)
  local lastStates1 = nn.SelectTable(1)(lastStates)
  local lastStates2 = nn.SelectTable(2)(lastStates)
  local intoActor = nn.JoinTable(2)({lastStates1,lastStates2})
--  intoActor = nn.BlockGradientLayer(params.batch_size,1024)(intoActor)
  --intoActor = nn.PrintLayer("intoActor")(intoActor)
  local inActor = nn.Linear(2*params.rnn_size,NUMBER_OF_ANSWER_OPTIONS)(intoActor)
  --inActor = nn.PrintLayer("inActor")(inActor)
  withActor = nn.LogSoftMax()(inActor)
  --withActor = nn.PrintLayer("withActor")(withActor)


  local model = nn.gModule({inputWords}, {reader2,withActor})
  model = transfer_data(model)
  model:getParameters():uniform(-params.init_weight, params.init_weight)


  return model
end




function neatQA.createAnswerNetworkWithMemoryAttention()
  local prev_c_table = nn.Identity()
  local lastState = nn.Identity()
  local prev_c_join = nn.JoinTable(2)(prev_c_table)
  local attention = nn.Linear(params.rnn_size,params.rnn_size)(nn.View(-1,params.rnn_size)(prev_c_join))
  attention = nn.View(params.batch_size,-1)(attention)
  local attention_sum = nn.Tanh()(nn.AddScalar()({attention}))
  attention_sum = nn.View(-1,params.rnn_size)(attention_sum)
  local attention_score = nn.Linear(params.rnn.size, 1)(attention_sum)
  attention_score = nn.View(batch_size,-1)(attention_score)
  attention_score = nn.SoftMax(2)(attention_score)
  attention_score = nn.View(batch_size, 1, -1)(attention_score)

  prev_c_join = nn.View(params.batch_size, -1, rnn_size)(prev_c_join)
  local prev_c = nn.View(batch_size, rnn_size)(nn.MM(false,false)({attention_score, prev_c_join}))
  crash()  
end

function neatQA.createAnswerNetwork()
  if neatQA.ACCESS_MEMORY then
    return neatQA.createAnswerNetworkWithMemoryAttention()
  else
    return neatQA.createSimpleAnswerNetwork()
  end
end

function buildDeepLSTMReader()
-- local model = nn.Sequential()
  local input = nn.Identity()()
  local embeddings = nn.Dropout(0.2)(nn.LookupTableMaskZero(params.vocab_size,params.embeddings_dimensionality)(input))
  local lastState = nn.Identity()()
  local lastCell = nn.Identity()()
  
  
  
end

function neatQA.setup()
  print("Creating a RNN LSTM network.")
  --crash()

  -- initialize data structures
  model.sR = {}
  model.dsR = {}
--  model.dsA = {}
  model.start_sR = {}
  for j = 0, params.seq_length do
    model.sR[j] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.dsR[1] = transfer_data(torch.zeros(params.rnn_size))
  model.dsR[2] = transfer_data(torch.zeros(params.rnn_size))
  if neatQA.ALSO_DO_LANGUAGE_MODELING then
    model.dsR[3] = transfer_data(torch.zeros(params.vocab_size)) -- NOTE actually will later have different size
  end

--  model.dsA[1] = transfer_data(torch.zeros(params.rnn_size))
--  model.dsA[2] = transfer_data(torch.zeros(params.rnn_size))
--  model.dsA[3] = transfer_data(torch.zeros(params.vocab_size)) -- NOTE actually will later have different size

  

  reader_c ={}
  reader_h = {}

  reader_output = {}

  if neatQA.number_of_LSTM_layers == 1 then
    reader_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero() 
    reader_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()
  else
    reader_c[0] = {}
    reader_h[0] = {}
    for layer=1,neatQA.number_of_LSTM_layers do
      table.insert(reader_c[0], torch.CudaTensor(params.batch_size, params.rnn_size):zero())
      table.insert(reader_h[0], torch.CudaTensor(params.batch_size, params.rnn_size):zero())
    end
  end


--  neatQA.answerCriterion = nn.ClassNLLCriterion()


  neatQA.criterionDerivative = torch.DoubleTensor(params.batch_size, NUMBER_OF_ANSWER_OPTIONS) 


  attention_decisions = {}
  attention_scores = {}
  baseline_scores = {}
  for i=1, params.seq_length do
     attention_decisions[i] = torch.CudaTensor(params.batch_size)
     attention_scores[i] = torch.CudaTensor(params.batch_size,1)
     baseline_scores[i] = torch.CudaTensor(params.batch_size,1)
  end

  probabilityOfChoices = torch.FloatTensor(params.batch_size)
  totalAttentions = torch.FloatTensor(params.batch_size) -- apparently using CudaTensor would cause a noticeable slowdown...?!
  nll = torch.FloatTensor(params.batch_size)

  attention_inputTensors = {}


  ones = torch.ones(params.batch_size):cuda()
  rewardBaseline = 0







   -- build alternative graph here
   twoLayerLSTMGraph = buildDeepLSTMReader() 


   --  local reader_core_network
    -- reader_core_network = autoencoding.create_network(neatQA.ALSO_DO_LANGUAGE_MODELING, true, true)

     neatQA.fullModel = neatQA.buildNeuralGraph()

     attentionNetwork = attention.createAttentionNetwork() --createAttentionNetwork()



     -- READER
     --paramxR, paramdxR = reader_core_network:getParameters()

     -- ACTOR
--     paramxA, paramdxA = actor_core_network:getParameters()

--since later using momentum
--paramdxA:zero()
--paramdxR:zero()



     -- ATTENTION
--     local attentionNetwork = attention.createAttentionNetwork()
--     paramxRA, paramdxRA = attentionNetwork:getParameters()


  if neatQA.INITIALIZE_FROM_NEAT then
    if LOAD then
       crash()
    end


      print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)
-- TODO add params
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, SparamxRA, SparamdxRA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load(BASE_DIRECTORY.."/model-"..fileToBeLoaded, "binary"))
  -- print(SparamxR)
 --     paramxR, paramdxR = reader_core_network:parameters()
 -- print(paramxR)

      paramxF, paramdxF = neatQA.fullModel:parameters()

     print(SparamxR)
     print(paramxF)
     print(paramxAltR)

--     paramxR[1]:set(paramxR[1])
     for i=2,10001 do
   --    paramxR[1][i]:copy(SparamxR[1][i-1])

       paramxF[1][i]:copy(SparamxR[1][i-1])
       paramdxF[1][i]:copy(SparamdxR[1][i-1])
     end
     for i=2, #SparamxR do
     --   paramxR[i]:set(SparamxR[i])  
        paramxF[i]:set(SparamxR[i])  
        paramdxF[i]:set(SparamdxR[i])   
     end
     print("Finished loading from NEAT")
     --print(paramxR)
     print(paramxF)
     print(paramdxF)
--   crash() 

  elseif LOAD then

     print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, SparamxRA, SparamdxRA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load(BASE_DIRECTORY.."/model-"..fileToBeLoaded, "binary"))

    if SparamxB == nil and USE_BIDIR_BASELINE and DO_TRAINING and IS_CONTINUING_ATTENTION then
        print("962 no baseline in saved file")
        crash()
    end

    print(params2)


------
--     local reader_core_network
  --   reader_core_network = autoencoding.create_network(false, true, true)
crash()
     -- LOAD PARAMETERS
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()
     for j=1, #SparamxR do
           reader_network_params[j]:set(SparamxR[j])
           reader_network_gradparams[j]:set(SparamxR[j])
     end
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()


     -- CLONE
--[[     readerRNNs = {}
     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network) --nn.MaskZero(g_clone(reader_core_network),1)
     end]]

------
     -- ACTOR
     
     --actor_core_network = neatQA.createAnswerNetwork()
     -- ATTENTION

     att_network_params, network_gradparams = attentionNetwork:parameters()
     if params.ATTENTION_WITH_EMBEDDINGS then
        if not IS_CONTINUING_ATTENTION then
           att_network_params[1]:set(reader_network_params[1])
           print("Using embeddings from the reader")
        else
           print("Not using embeddings from the reader because continuing attention")
        end
     end


     if USE_BIDIR_BASELINE and DO_TRAINING then
          setupBidirBaseline(reader_network_params, SparamxB, SparamdxB)
     end



     if IS_CONTINUING_ATTENTION then
         network_params, network_gradparams = attentionNetwork:parameters()

         for j=1, #SparamxRA do
            network_params[j]:set(SparamxRA[j])
            network_gradparams[j]:set(SparamdxRA[j])
         end


         print("Got attention network from file")
     else
         print("NOTE am not using the attention network from the file")
     end



     -- for safety zero initialization when later using momentum
     print("Sequences read by model "..sentencesRead)

     reader_c[0] = readerCStart
     reader_h[0] = readerHStart
   end


     paramxRA, paramdxRA = attentionNetwork:getParameters()
     paramxF, paramdxF = neatQA.fullModel:getParameters()

 assert(paramxF:size()[1] == paramdxF:size()[1])



local xF, dxF = neatQA.fullModel:parameters()

for i=1,#dxF do
dxF[i]:zero()
end




     paramdxRA:zero()
paramdxF:zero()














   neatQA.readerCFinal = transfer_data(torch.zeros(params.batch_size,params.rnn_size))
   neatQA.readerHFinal = transfer_data(torch.zeros(params.batch_size,params.rnn_size))
 


   vectorOfLengths = torch.LongTensor(params.batch_size)
--   neatQA.maximalLengthOccurringInInput = {0}
end


function neatQA.emptyCHTables()
  local readerCInitial = reader_c[0]
  local readerHInitial = reader_h[0]
  reader_c = {}
  reader_h = {}
  reader_c[0] = readerCInitial
  reader_h[0] = readerHInitial
end

function neatQA.fp(corpus, startIndex, endIndex)
  -- since we want the length to be bounded by the input for attention, remove all the later entries
  neatQA.emptyCHTables()

  probabilityOfChoices:fill(1)
  totalAttentions:fill(params.ATTENTION_VALUES_BASELINE)
  neatQA.inputTensors = auxiliary.buildInputTensorsQA(corpus, startIndex, endIndex, vectorOfLengths, neatQA.maximalLengthOccurringInInput)
  neatQA.inputTensorsTables = auxiliary.toUnaryTables(neatQA.inputTensors)

--  print("49412")
--print(neatQA.maximalLengthOccurringInInput)
--print(#neatQA.inputTensors)
--print(#neatQA.inputTensorsTables)

  neatQA.answerTensors =  qa.buildAnswerTensor(corpus, startIndex, endIndex)


  neatQA.fullOutput = neatQA.fullModel:forward(neatQA.inputTensorsTables)
  --print(neatQA.fullOutput)
--  crash()
--  print(neatQA.inputTensorsTables)
  --print(neatQA.altOutput)
  --print(neatQA.altOutput[5][3])
--  crash()
  --print("200")
  --print(neatQA.answerTensors)


  print("40  "..neatQA.maximalLengthOccurringInInput[1])




--  print("&&&&")
--  print(neatQA.fullOutput[2][5][1])
--  print(neatQA.fullOutput[2]:size())
--  print("????")

  -- is it necessary to clone this?

  -- cannot simply use ClassNLLCriterion, as we need the scores for the various batch items
  for i=1, params.batch_size do
    nll[i] = - neatQA.fullOutput[2][i][neatQA.answerTensors[i]]
  end


  meanNLL = 0.95 * meanNLL + 0.05 * nll:mean()

  return nll, neatQA.fullOutput[2]
end


function neatQA.bp(corpus, startIndex, endIndex)



  paramdxF:mul(params.lr_momentum / (1-params.lr_momentum))

  reset_ds()
  
  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true
  
  if params.lr > 0 and (true or train_autoencoding) then --hrhr
    -- c, h
    derivativeFromCriterion = neatQA.criterionDerivative
    derivativeFromCriterion:zero()
    for i=1, params.batch_size do
      derivativeFromCriterion[i][neatQA.answerTensors[i]] = -1
    end

    local actorInputs
    if neatQA.ACCESS_MEMORY then
       actorInputs = {neatQA.readerCFinal, neatQA.readerHFinal, reader_c}
    else
      actorInputs = {neatQA.readerCFinal, neatQA.readerHFinal}
    end





      model.dsFull = {}
      model.dsFull[1] = {{},{}}
      for i = 1,neatQA.maximalLengthOccurringInInput[1] do
        model.dsFull[1][1][i] = {}
        model.dsFull[1][1][i][1] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()
        model.dsFull[1][1][i][2] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()
        model.dsFull[1][1][i][3] = torch.CudaTensor(params.batch_size,params.vocab_size):zero()
      end
      model.dsFull[1][2][1] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()
      model.dsFull[1][2][2] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()
      model.dsFull[1][2][3] = torch.CudaTensor(params.batch_size,params.vocab_size):zero()

      model.dsFull[2] = derivativeFromCriterion:cuda()
      neatQA.fullGradient = neatQA.fullModel:backward(neatQA.inputTensorsTables, model.dsFull)

for m=1,0 do
i = math.random(2048)
j = math.random(100)
print("@@@")
print(dxF[2][i][j])
print(xF[2][i][j])

end
for m=1,0 do
i = math.random(100)
j = math.random(100)
print("%%%%")
print(dxF[8][i][j])
print(dxA[1][i][j])
print(xF[8][i][j])
print(xA[1][i][j])

end

for m=1,0 do
 i = math.random(512)
 j = math.random(60)
 print(",,,")
-- print(neatQA.fullOutput[1][2])
 print(neatQA.fullOutput[1][2][1][j][i])
 print(neatQA.readerCFinal[j][i])
end


--      print("851  "..paramdxF:norm().."  "..params.max_grad_norm)
      auxiliary.clipGradients(paramdxF)
 
--      print("851  "..paramdxF:norm().."  "..params.max_grad_norm)
-- print(paramxF:size())
-- print(paramdxF:size())
if true then
      auxiliary.updateParametersWithMomentum(paramxF,paramdxF)
--      auxiliary.updateParametersWithMomentum(paramxR,paramdxR)
end
--[[      paramdxAltR:mul((1-params.lr_momentum))
      paramxAltR:add(paramdxAltR:mul(- 1 * params.lr))
      paramdxAltR:mul(1 / (- 1 * params.lr)) -- is this really better than cloning before multiplying?

     
      paramdxR:mul((1-params.lr_momentum))
      paramxR:add(paramdxR:mul(- 1 * params.lr))
      paramdxR:mul(1 / (- 1 * params.lr)) -- is this really better than cloning before multiplying?


      paramdxA:mul((1-params.lr_momentum))
      paramxA:add(paramdxA:mul(- 1 * params.lr))
      paramdxA:mul(1 / (- 1 * params.lr)) -- is this really better than cloning before multiplying?
]]

  end
  
  
  if params.lr_att > 0 and (true and train_attention_network) then
--     print("346")
  --   print(nll)
    -- print(params.TOTAL_ATTENTIONS_WEIGHT)
    -- print(totalAttentions)
     local reward = torch.add(nll,params.TOTAL_ATTENTIONS_WEIGHT,totalAttentions):cuda() -- gives the reward for each batch item


     local rewardDifference = reward:clone():cuda():add(-rewardBaseline, ones) 

     local baselineScores
     if USE_BIDIR_BASELINE then
        bidir_baseline_gradparams:zero()
        local inputsWithActions = {}
        for i=1, params.seq_length do
            inputsWithActions[i] = joinTable[i]:forward({neatQA.inputTensors[i]:view(-1,1), attention_decisions[i]:view(-1,1)})
        end
        baselineScores = bidir_baseline:forward(inputsWithActions)
        for i=1, params.seq_length do
          local target
          if i == params.seq_length then
             target = reward:view(-1,1)
         else
             target = baselineScores[i+1]:view(-1,1)
          end

             if false and i == params.seq_length and torch.uniform() > 0.9 then
               print("TARGET ")
               print(target)

               print("SCORES ")
               print(baselineScores[i])
             end
 



          bidirBaseline.criterion_gradients[i]:copy(baseline_criterion:backward(baselineScores[i]:view(-1,1), target))
          if i == params.seq_length then
             --print(baseline_criterion:backward(baselineScores[i]:view(-1,1), target))
          end
       end
       bidir_baseline:backward(inputsWithActions, bidirBaseline.criterion_gradients)

     local norm_dwB = bidir_baseline_gradparams:norm()
     if norm_dwB > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / norm_dwB
        bidir_baseline_gradparams:mul(shrink_factor)
     end
       bidir_baseline_params:add(bidir_baseline_gradparams:mul(- 0.1))
     end

     rewardBaseline = 0.8 * rewardBaseline + 0.2 * torch.sum(reward) * 1/params.batch_size
     for i = params.seq_length, 1, -1 do
        rewardDifferenceForI = rewardDifference
        if USE_BIDIR_BASELINE then
            if i>1 then
              rewardDifferenceForI = baselineScores[i-1]:clone():add(-1,reward):mul(-1)
if false and torch.uniform() > 0.9 and i == 25 then
 print(i.."   "..(torch.mean(torch.abs(rewardDifference) - torch.abs(rewardDifferenceForI))))
end

            end
        end

       

        local whatToMultiplyToTheFinalDerivative = torch.CudaTensor(params.batch_size)
        local attentionEntropyFactor =  torch.CudaTensor(params.batch_size)
        for j=1,params.batch_size do
          attentionEntropyFactor[j] =  params.ENTROPY_PENALTY * (math.log(attention_scores[i][j][1]) - math.log(1 - attention_scores[i][j][1]))
           if attention_decisions[i][j] == 0 then
               whatToMultiplyToTheFinalDerivative[j] = -1 / (1 - attention_scores[i][j][1])
           else
               whatToMultiplyToTheFinalDerivative[j] = 1 / (attention_scores[i][j][1])
           end
        end
        local factorsForTheDerivatives =  rewardDifferenceForI:clone():cmul(whatToMultiplyToTheFinalDerivative)
        factorsForTheDerivatives:add(attentionEntropyFactor)

     local inputTensor
     if i>1 and USE_PREDICTION_FOR_ATTENTION then
        inputTensor = attention_inputTensors[i]
     else 
         inputTensor = qa.inputTensors[i-1]
     end


        local tmp = attentionNetworks[i]:backward({inputTensor, reader_c[i-1]},factorsForTheDerivatives)
     end
     local norm_dwRA = paramdxRA:norm()
     if norm_dwRA > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / norm_dwRA
        paramdxRA:mul(shrink_factor)
     end
     assert(norm_dwRA == norm_dwRA)
  
     -- MOMENTUM
     paramdxRA:mul((1-params.lr_momentum))
     paramxRA:add(paramdxRA:mul(- 1 * params.lr_att))
     paramdxRA:mul(1 / (- 1 * params.lr_att)) -- is this really better than cloning before multiplying?
  end
  
  
end





function neatQA.printStuff(perp, actor_output, since_beginning, epoch, numberOfWords)

            print("+++++++ "..perp[1]..'  '..meanNLL)
             print(epoch.."  "..readChunks.corpusReading.currentFile..
               '   since beginning = ' .. since_beginning .. ' mins.')  
            print(experimentNameOut)
            print(params) 

            local correct = 0.0
            local incorrect = 0.0  
            for l = 1, params.batch_size do
               print("batch index "..l)
               --print("PERP "..perp[l])
               --print(neatQA.answerTensors[l])
               --print(qa.getFromAnswer(readChunks.corpus,l,1))
               --print("todo make sure the right numbers are used for the answers (numbers vs. numbersToEntityIDs)")
               
               local answerID = qa.getFromAnswer(readChunks.corpus,l,1)
               if answerID == nil then
                    print("463: answerID == nil")
                    answerID = 1
               end
               if  (math.random() < 0.001) then
                  auxiliary.deepPrint(neatQA.inputTensors, function (tens) return tens[l] end)


                  for j=2,params.seq_length do
                    if neatQA.inputTensors[j][l] == 0 then
                       break
                    end
                    local wordProbabilities = neatQA.fullOutput[1][1][j-1][3][l]
                    local predictedScoresLM, predictedTokensLM = torch.min( wordProbabilities,1)
                    io.write((readDict.chars[neatQA.inputTensors[j][l]]))--..'\n')
                    io.write(" \t "..readDict.chars[predictedTokensLM[1]])
                    io.write("  "..math.exp(-predictedScoresLM[1]))
                    io.write("  "..math.exp(-wordProbabilities[neatQA.inputTensors[j][l]]).."\n")
                  end


               end
--               print(readChunks.corpus[l].text)
               print("ANSW "..answerID)
               print("PROB FULL "..neatQA.fullOutput[2][l][answerID])
               print("PROB ALT  "..actor_output[l][answerID])
               local predictedScoreFull,predictedAnswerFull = torch.max(neatQA.fullOutput[2][l],1)
if false then
               local predictedScoreAlt,predictedAnswerAlt = torch.max(actor_output[l],1)
               print("PREDICTED  ALT  "..predictedAnswerAlt[1].." # "..predictedScoreAlt[1])

end
               print("PREDICTED  FULL "..predictedAnswerFull[1].." # "..predictedScoreFull[1])



--               local negSample = math.random(math.min(10, neatQA.fullOutput[2][l]:size()[1]))
  --             print("NEGATIVE EX PROB "..neatQA.fullOutput[2][l][negSample].." ("..negSample..")")
--               if (math.abs(neatQA.fullOutput[2][l][answerID]) <= math.abs(neatQA.fullOutput[2][l][negSample])) then
               if (predictedAnswerFull[1] == answerID) then
                 correct = correct + 1.0
               else
                 incorrect = incorrect + 1.0
               end
               --print(actor_output[l])
               --print("PERP "..math.exp(-actor_output[l][answerID]))

            end
            fileStats:write((numberOfWords/params.seq_length)..'\t'..perp[1]..'\n')
            fileStats:flush()
            print("APPROX PERFORMANCE  "..(correct / (correct + incorrect)))
            globalForExpOutput.accuracy = 0.95 * globalForExpOutput.accuracy + 0.05 * (correct / (correct+incorrect))

            print("Avg performance     "..(globalForExpOutput.accuracy))
--            globalForExpOutput.correct = globalForExpOutput.correct * 0.9
  --          globalForExpOutput.incorrect = globalForExpOutput.incorrect * 0.9
end




