neatQA = {}
assert(false)

neatQA.number_of_LSTM_layers = 1

neatQA.ALSO_DO_LANGUAGE_MODELING = false

neatQA.ACCESS_MEMORY = false

neatQA.DO_LENGTH_ADAPTIVE = false

-- input: recurrent state and memory cells
-- output: softmax layer
function neatQA.createSimpleAnswerNetwork()
  local model = nn.Sequential()
  model:add(nn.JoinTable(2))
  model:add(nn.Linear(2*params.rnn_size,NUMBER_OF_ANSWER_OPTIONS))
  model:add(nn.LogSoftMax())
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

function neatQA.setup()
  print("Creating a RNN LSTM network.")


  -- initialize data structures
  model.sR = {}
  model.dsR = {}
  model.dsA = {}
  model.start_sR = {}
  for j = 0, params.seq_length do
    model.sR[j] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.dsR[1] = transfer_data(torch.zeros(params.rnn_size))
  model.dsR[2] = transfer_data(torch.zeros(params.rnn_size))

  model.dsA[1] = transfer_data(torch.zeros(params.rnn_size))
  model.dsA[2] = transfer_data(torch.zeros(params.rnn_size))
  model.dsA[3] = transfer_data(torch.zeros(params.rnn_size)) -- NOTE actually will later have different size

  

  reader_c ={}
  reader_h = {}


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

   if not LOAD then
     -- READER
     local reader_core_network
     reader_core_network = autoencoding.create_network(neatQA.ALSO_DO_LANGUAGE_MODELING, true, true)
     paramxR, paramdxR = reader_core_network:getParameters()

     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network)
     end

     -- ACTOR
     actor_core_network = neatQA.createAnswerNetwork()
     paramxA, paramdxA = actor_core_network:getParameters()

--since later using momentum
paramdxA:zero()
paramdxR:zero()



     -- ATTENTION
     local attentionNetwork = attention.createAttentionNetwork()
     paramxRA, paramdxRA = attentionNetwork:getParameters()

     attentionNetworks = {}

     for i=1,params.seq_length do
        attentionNetworks[i] = g_clone(attentionNetwork)
     end
  elseif true then

     print("LOADING MODEL AT ".."/disk/scratch2/s1582047/model-"..fileToBeLoaded)
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, SparamxRA, SparamdxRA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load("/disk/scratch2/s1582047/model-"..fileToBeLoaded, "binary"))

    if SparamxB == nil and USE_BIDIR_BASELINE and DO_TRAINING and IS_CONTINUING_ATTENTION then
        print("962 no baseline in saved file")
        crash()
    end

    print(params2)


------
     local reader_core_network
     reader_core_network = autoencoding.create_network(false, true, true)

     -- LOAD PARAMETERS
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()
     for j=1, #SparamxR do
           reader_network_params[j]:set(SparamxR[j])
           reader_network_gradparams[j]:set(SparamxR[j])
     end
     paramxR, paramdxR = reader_core_network:getParameters()
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()


     -- CLONE
     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network) --nn.MaskZero(g_clone(reader_core_network),1)
     end

------
     -- ACTOR
     
     actor_core_network = neatQA.createAnswerNetwork()
     actor_network_params, actor_network_gradparams = actor_core_network:parameters()
     for j=1, #SparamxA do
           actor_network_params[j]:set(SparamxA[j])
           actor_network_gradparams[j]:set(SparamdxA[j])
     end
     paramxA, paramdxA = actor_core_network:getParameters()


     -- ATTENTION

     local attentionNetwork = attention.createAttentionNetwork() --createAttentionNetwork()
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

     paramxRA, paramdxRA = attentionNetwork:getParameters()
     attentionNetworks = {}
     for i=1,params.seq_length do
        attentionNetworks[i] = g_clone(attentionNetwork)
     end


     -- for safety zero initialization when later using momentum
     paramdxRA:zero()
paramdxA:zero()
paramdxR:zero()

     print("Sequences read by model "..sentencesRead)

     reader_c[0] = readerCStart
     reader_h[0] = readerHStart
   end

   neatQA.readerCFinal = transfer_data(torch.zeros(params.batch_size,params.rnn_size))
   neatQA.readerHFinal = transfer_data(torch.zeros(params.batch_size,params.rnn_size))
 


   vectorOfLengths = torch.LongTensor(params.batch_size)
   neatQA.maximalLengthOccurringInInput = {0}
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
  neatQA.answerTensors =  qa.buildAnswerTensor(corpus, startIndex, endIndex)

  --print("200")
  --print(neatQA.answerTensors)


  if(false) then
    print("..")
    print(neatQA.inputTensors[1])
    print(neatQA.inputTensors[2])
    print(neatQA.inputTensors[3])
    print(neatQA.inputTensors[4])
    print(neatQA.inputTensors[5])
    print(neatQA.inputTensors[6])
    print(neatQA.inputTensors[7])
    print("==")
    print(neatQA.answerTensors)
    print("..")
    print("READER")
  end
  --print("218")
  --print(neatQA.inputTensors)
  print("40  "..neatQA.maximalLengthOccurringInInput[1])
  for i=1, neatQA.maximalLengthOccurringInInput[1] do
     --print(i)
     --print(neatQA.inputTensors[i])
     local inputTensor = neatQA.inputTensors[i]
--      local attendedInputTensor, probability = hardAttention.makeAttentionDecisions(i, inputTensor)
--
--      TODO the multi-layer implementation is not done yet. probably better to put it inside the graph
     --print("297")
     --print(i)
     --print(reader_h[i-1])
     if neatQA.number_of_LSTM_layers == 1 then
      reader_c[i], reader_h[i] = unpack(readerRNNs[i]:forward({inputTensor, reader_c[i-1], reader_h[i-1]}))
     else
       local inputTensorsForFP = {inputTensor}
       for layer=1,neatQA.number_of_LSTM_layers do
          table.insert(inputTensorsForFP,reader_c[i-1][layer])
          table.insert(inputTensorsForFP,reader_h[i-1][layer])
       end
--       reader_c[i], reader_h[i] = unpack(
       local outputTensorsForFP = readerRNNs[i]:forward(inputTensorsForFP)
       for layer=1,neatQA.number_of_LSTM_layers do
         reader_c[i][layer] = outputTensorsForFP[(layer-1)*2+1]
         reader_h[i][layer] = outputTensorsForFP[(layer-1)*2+2]
       end
     end
      --print(inputTensor)
      --print(reader_c[i])
  --   totalAttentions:add(attention_decisions[i])
  end

  if(false) then
    print(reader_c[params.seq_length])
  end
  --print("lengths")
  --print(vectorOfLengths)
  if neatQA.DO_LENGTH_ADAPTIVE then
    for i=1, params.batch_size do
      neatQA.readerCFinal[i]:copy(reader_c[vectorOfLengths[i]][i])
      neatQA.readerHFinal[i]:copy(reader_h[vectorOfLengths[i]][i])
    end
  else
    neatQA.readerCFinal = reader_c[neatQA.maximalLengthOccurringInInput[1]]
    neatQA.readerHFinal = reader_h[neatQA.maximalLengthOccurringInInput[1]]   
  end
  if neatQA.ACCESS_MEMORY then
     actor_output = actor_core_network:forward({neatQA.readerCFinal, neatQA.readerHFinal, reader_c}):float()
  else
     actor_output = actor_core_network:forward({neatQA.readerCFinal, neatQA.readerHFinal}):float()
  end

  -- is it necessary to clone this?

  -- cannot simply use ClassNLLCriterion, as we need the scores for the various batch items
  
  --nll = neatQA.answerCriterion:forward(actor_output, neatQA.answerTensors)
  --print(nll) 
  --nll = nll:clone()

-- fill in
  if(false) then 
    print(actor_output)
  end
  --print("263")
  --print(neatQA.answerTensors)
  for i=1, params.batch_size do
    --print(neatQA.answerTensors[i])
    --print(actor_output[i])
    nll[i] = - actor_output[i][neatQA.answerTensors[i]]
  end

  --print(nll)
  meanNLL = 0.95 * meanNLL + 0.05 * nll:mean()

  return nll, actor_output
end


function neatQA.bp(corpus, startIndex, endIndex)
  
   -- MOMENTUM 
  paramdxR:mul(params.lr_momentum / (1-params.lr_momentum))
--:zero()
  paramdxA:mul(params.lr_momentum / (1-params.lr_momentum))
--:zero()
  paramdxRA:mul(params.lr_momentum / (1-params.lr_momentum))
  
  
  reset_ds()


  
  
  
--  local inputTensors = auxiliary.buildInputTensorsQA(corpus, startIndex, endIndex)
  
  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true
  
  if params.lr > 0 and (true or train_autoencoding) then --hrhr
    -- c, h
    derivativeFromCriterion = neatQA.criterionDerivative
    derivativeFromCriterion:zero()
    for i=1, params.batch_size do
      derivativeFromCriterion[i][neatQA.answerTensors[i]] = -1
    end
--    print(derivativeFromCriterion)

   
    --neatQA.answerCriterion:backward(actor_output, neatQA.answerTensors)

    local actorInputs
    if neatQA.ACCESS_MEMORY then
       actorInputs = {neatQA.readerCFinal, neatQA.readerHFinal, reader_c}
    else
      actorInputs = {neatQA.readerCFinal, neatQA.readerHFinal}
    end
    local actorGradient = actor_core_network:backward(actorInputs, derivativeFromCriterion)

--    print(tmp[1])
  --  print(tmp[2])

   -- if TRAIN_AUTOENCODER then
   --   model.dsR[1]:copy(actorGradient[1])
   --   model.dsR[2]:copy(actorGradient[2])
   --else
      model.dsR[1]:zero()
      model.dsR[2]:zero()
   --end

     if (not neatQA.DO_LENGTH_ADAPTIVE) then
       model.dsR[1]:copy(actorGradient[1])
       model.dsR[2]:copy(actorGradient[2])
     end
  
      for i = neatQA.maximalLengthOccurringInInput[1], 1, -1 do
          if neatQA.DO_LENGTH_ADAPTIVE then
             for j = 1, params.batch_size do
               if vectorOfLengths[j] == i then
                -- TODO get the entry
                 model.dsR[1][j]:add(actorGradient[1][j])
                 model.dsR[2][j]:add(actorGradient[2][j])
           --   elseif i == neatQA.maximalLengthOccurringInInput[1] and vectorOfLengths[j] < i then
           --      model.dsR[1][j]:zero()
           --      model.dsR[2][j]:zero()
                end
             end
          end

          -- from the attention
          if neatQA.ACCESS_MEMORY then 
             model.dsR[1]:add(actorGradient[3][i])
          end

          local inputTensor = neatQA.inputTensors[i]
   --       inputTensor= torch.cmul(neatQA.inputTensors[i], attention_decisions[i])
  
          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
          --local derr = transfer_data(torch.ones(1))
 

 
          local tmp = readerRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                        model.dsR)
--          print(tmp[2])
  --        print(tmp[3])
          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          cutorch.synchronize()
      end
  
      model.norm_dwR = paramdxR:norm()
      if model.norm_dwR > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwR
          paramdxR:mul(shrink_factor)
      end
  
      model.norm_dwA = paramdxA:norm()
      if model.norm_dwA > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwA
          paramdxA:mul(shrink_factor)
      end
  

      paramdxR:mul((1-params.lr_momentum))
      paramxR:add(paramdxR:mul(- 1 * params.lr_att))
      paramdxR:mul(1 / (- 1 * params.lr_att)) -- is this really better than cloning before multiplying?

      paramdxA:mul((1-params.lr_momentum))
      paramxA:add(paramdxA:mul(- 1 * params.lr_att))
      paramdxA:mul(1 / (- 1 * params.lr_att)) -- is this really better than cloning before multiplying?


 
--      paramxR:add(paramdxR:mul(-params.lr))
  --    paramxA:add(paramdxA:mul(-params.lr))
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
               
               auxiliary.deepPrint(neatQA.inputTensors, function (tens) return tens[l] end)
--               print(readChunks.corpus[l].text)
               print("ANSW "..answerID)
               print("PROB "..actor_output[l][answerID])
               local negSample = math.random(math.min(10, actor_output[l]:size()[1]))
               print("NEGATIVE EX PROB "..actor_output[l][negSample].." ("..negSample..")")
               if (math.abs(actor_output[l][answerID]) <= math.abs(actor_output[l][negSample])) then
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




