-- presumably superseded by neat-qa-Unrolled-Attention, which can deal both with random attention and with attention networks

-- This is good for training QA with fixed random attention, and attention coming from external numerical input.
-- However, it does not implement an attention network. For that, use neat-qa-Unrolled-Attention.
-- It would be much better if the two files were unified so that only fp and bp were split. because the remainder is almost totally the same.

assert(false)


neatQA = {}
neatQA.__name = "neat-qa-UNDO-THE-CHANGES"

print("LOADING neat-qa-UNDO-THE-CHANGES.lua")

neatQA.number_of_LSTM_layers = 1

neatQA.ALSO_DO_LANGUAGE_MODELING = false --true

neatQA.ACCESS_MEMORY = true--false--true

neatQA.INITIALIZE_FROM_NEAT = true

neatQA.DO_BIDIRECTIONAL_MEMORY = false


neatQA.USE_ATTENTION_NETWORK = false

print(neatQA)


require('nn.RecursorMod')
require('nn.SequencerMod')
require('nn.PrintLayer')
require('nn.BlockGradientLayer')
require('qaAttentionAnswerer')
require('recurrentNetworkOnSequence')

--print(recurrentNetworkOnSequence)
--print(RecurrentNetworkOnSequence)
--crash()

--assert(not neatQA.ACCESS_MEMORY)

-- input: recurrent state and memory cells
-- output: softmax layer
function neatQA.createSimpleAnswerNetwork()
  local model = nn.Sequential()
  model:add(nn.JoinTable(2))
  model:add(nn.Linear(2*params.rnn_size,NUMBER_OF_ANSWER_OPTIONS))
  model:add(nn.LogSoftMax())
  assert(false, "This should presumably become a GPU network")
  return model
end



function neatQA.createAnswerNetwork()
  if neatQA.ACCESS_MEMORY then
    return qaAttentionAnswerer.createAnswerNetworkWithMemoryAttention()
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



  neatQA.criterionDerivative = torch.DoubleTensor(params.batch_size, NUMBER_OF_ANSWER_OPTIONS) 


  attention_decisions = {}
  attention_scores = {}
  baseline_scores = {}
  attended_input_tensors = {}
  for i=1, params.seq_length do
     attention_decisions[i] = torch.CudaTensor(params.batch_size)
     attention_scores[i] = torch.CudaTensor(params.batch_size,1)
     baseline_scores[i] = torch.CudaTensor(params.batch_size,1)
     attended_input_tensors[i] = torch.CudaTensor(params.batch_size,1)
  end

  probabilityOfChoices = torch.FloatTensor(params.batch_size)
  totalAttentions = torch.FloatTensor(params.batch_size) -- apparently using CudaTensor would cause a noticeable slowdown...?!
  nll = torch.FloatTensor(params.batch_size)

  attention_inputTensors = {}


  ones = torch.ones(params.batch_size):cuda()
  rewardBaseline = 0







   -- build alternative graph here
   --twoLayerLSTMGraph = buildDeepLSTMReader() 


     local reader_core_network
     reader_core_network = autoencoding.create_network(neatQA.ALSO_DO_LANGUAGE_MODELING, true, true)

     actor_core_network = neatQA.createAnswerNetwork()

     attentionNetwork = attention.createAttentionNetwork() --createAttentionNetwork()

if false then
     local paramsAct = actor_core_network:parameters()
     local paramsFull = neatQA.fullModel:parameters()
     for i=1,#paramsAct do
       paramsFull[7+i]:copy(paramsAct[i])
     end
end

  if (not LOAD) and neatQA.INITIALIZE_FROM_NEAT then
    if LOAD then
       crash()
    end


      print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)
-- TODO add params
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, SparamxRA, SparamdxRA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load(BASE_DIRECTORY.."/model-"..fileToBeLoaded, "binary"))
   print(SparamxR)
      paramxR, paramdxR = reader_core_network:parameters()
  print(paramxR)

     for i=2,10001 do
       paramxR[1][i]:copy(SparamxR[1][i-1])
     end
     for i=2, #paramxR do
        paramxR[i]:set(SparamxR[i])  
     end
     print("Finished loading from NEAT")
     print(paramxR)


--   crash() 

  elseif LOAD then

     print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, SparamxRA, SparamdxRA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load(BASE_DIRECTORY.."/model-"..fileToBeLoaded, "binary"))

    if SparamxB == nil and USE_BIDIR_BASELINE and DO_TRAINING and IS_CONTINUING_ATTENTION then
        print("962 no baseline in saved file")
        crash()
    end

    print(params2)

     reader_network_params, reader_network_gradparams = reader_core_network:parameters()

    if (#reader_network_params ~= #SparamxR) then
       print("WARNING")
       print(SparamdxR)
       print(reader_network_params)
    end
    -- LOAD PARAMETERS
     for j=1, #reader_network_params do
           reader_network_params[j]:set(SparamxR[j])
           reader_network_gradparams[j]:set(SparamxR[j])
     end
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()

------
     -- ACTOR
     
     actor_network_params, actor_network_gradparams = actor_core_network:parameters()
     for j=1, #SparamxA do
           actor_network_params[j]:set(SparamxA[j])
           actor_network_gradparams[j]:set(SparamdxA[j])
     end
     

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


if neatQA.DO_BIDIRECTIONAL_MEMORY then
   backwards_network = RecurrentNetworkOnSequence.new(params.rnn_size, reader_core_network:parameters())
end



     paramxR, paramdxR = reader_core_network:getParameters()
     paramxA, paramdxA = actor_core_network:getParameters()
     paramxRA, paramdxRA = attentionNetwork:getParameters()

--auxiliary.printMemory("262")

     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network)
        print(i)
--auxiliary.printMemory("270")

     end

--auxiliary.printMemory("270")

     attentionNetworks = {}
     for i=1,params.seq_length do
        attentionNetworks[i] = g_clone(attentionNetwork)
        print(i)
--auxiliary.printMemory("270")

     end


--auxiliary.printMemory("277")







     paramdxRA:zero()
paramdxA:zero()
paramdxR:zero()














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
--auxiliary.printMemory("323")

  -- since we want the length to be bounded by the input for attention, remove all the later entries
  neatQA.emptyCHTables()

  probabilityOfChoices:fill(1)
  totalAttentions:fill(params.ATTENTION_VALUES_BASELINE)
  neatQA.inputTensors = auxiliary.buildInputTensorsQA(corpus, startIndex, endIndex, vectorOfLengths, neatQA.maximalLengthOccurringInInput)
  neatQA.inputTensorsTables = auxiliary.toUnaryTables(neatQA.inputTensors)

  neatQA.answerTensors =  qa.buildAnswerTensor(corpus, startIndex, endIndex)

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
  print("40  "..neatQA.maximalLengthOccurringInInput[1])
  for i=1, neatQA.maximalLengthOccurringInInput[1] do
     local inputTensor = neatQA.inputTensors[i]
--auxiliary.printMemory("352 "+i)
--print(attended_input_tensors[i])
-- somewhat unfortunately, there is a mismatch in the dimensionality expect of attention_decisions by an attention nngraph and by hardAttention.makeAttentionDecisions
attention_decisions[i] = attention_decisions[i]:view(-1)
     attended_input_tensors[i], _ = hardAttention.makeAttentionDecisions(i, inputTensor)
--attended_input_tensors[i] = attended_input_tensors[i]:view(params.batch_size,1)
attention_decisions[i] = attention_decisions[i]:view(params.batch_size,1)
--print(attended_input_tensors[i])

     reader_c[i], reader_h[i], reader_output[i] = unpack(readerRNNs[i]:forward({attended_input_tensors[i], reader_c[i-1], reader_h[i-1]}))
  end
  if(false) then
    print(reader_c[params.seq_length])
  end
  neatQA.readerCFinal = reader_c[neatQA.maximalLengthOccurringInInput[1]]
  neatQA.readerHFinal = reader_h[neatQA.maximalLengthOccurringInInput[1]]



  if neatQA.ACCESS_MEMORY then
     actor_output = actor_core_network:forward({neatQA.readerCFinal, neatQA.readerHFinal, reader_c}):float()
  else
     actor_output = actor_core_network:forward({neatQA.readerCFinal, neatQA.readerHFinal}):float()
  end
-- fill in
  if(false) then 
    print(actor_output)
  end
  for i=1, params.batch_size do
    nll[i] = - actor_output[i][neatQA.answerTensors[i]]
  end

if neatQA.DO_BIDIRECTIONAL_MEMORY then
  local cs, hs = backwards_network:fp(attended_input_tensors,neatQA.maximalLengthOccurringInInput[1])
  if true then
   for u =1,50 do
--     local i = math.random(60)
  --   local j = math.random(100)
     print("...")
     print(cs[5][u][u])
     print(reader_c[5][u][u])
   end
  end
end


  --print(nll)
  meanNLL = 0.95 * meanNLL + 0.05 * nll:mean()
--collectgarbage()

  return nll, actor_output
end


function neatQA.bp(corpus, startIndex, endIndex)
   --print("41011")
   --  print(paramdxA:norm())
 
   -- MOMENTUM 
  paramdxR:mul(params.lr_momentum / (1-params.lr_momentum))
  paramdxA:mul(params.lr_momentum / (1-params.lr_momentum))
  paramdxRA:mul(params.lr_momentum / (1-params.lr_momentum))

  reset_ds()

-- auxiliary.printMemory("407")
 
  
  
  
  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true
  
  if params.lr > 0 and (true or train_autoencoding) then --hrhr




 if neatQA.DO_BIDIRECTIONAL_MEMORY then
   print("161012")
   print(paramdxR:norm())
 end





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


    local actorGradient = actor_core_network:backward(actorInputs, derivativeFromCriterion:cuda())


if false then
      model.dsAltR = {}
      for i = 1,neatQA.maximalLengthOccurringInInput[1] do
        model.dsAltR[i] = {}
        model.dsAltR[i][1] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()
        model.dsAltR[i][2] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()
        model.dsAltR[i][3] = torch.CudaTensor(params.batch_size,params.vocab_size):zero()
      end
      model.dsAltR[neatQA.maximalLengthOccurringInInput[1]][1]:copy(actorGradient[1])
      model.dsAltR[neatQA.maximalLengthOccurringInInput[1]][2]:copy(actorGradient[2])

      neatQA.altGradient = neatQA.alternativeReader:backward(neatQA.inputTensorsTables, model.dsAltR)

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
      if true then --hullu
        for hullu=1,params.batch_size do
           model.dsFull[2][hullu][1] =  10
        end
      end
      neatQA.fullGradient = neatQA.fullModel:backward(neatQA.inputTensorsTables, model.dsFull)
end

      model.dsR[1]:copy(actorGradient[1])
      model.dsR[2]:copy(actorGradient[2])
 
      for i = neatQA.maximalLengthOccurringInInput[1], 1, -1 do
          if neatQA.ACCESS_MEMORY then 
             model.dsR[1]:add(actorGradient[3][i])
          end

          local inputTensor = attended_input_tensors[i]
 
          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]

 if neatQA.DO_BIDIRECTIONAL_MEMORY then
          print("52618 "..i.."  "..model.dsR[1]:norm().."  "..model.dsR[2]:norm())
 end
 
          local tmp = readerRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                        model.dsR)
          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          cutorch.synchronize()
      end




if false then
      for i=1,5 do
        local index = math.random(paramdxR:size(1))
        print("~~~~")
        print(paramdxR[index])
       -- print(paramdxF[index])
     --   print(paramdxAltR[index])
      end
end

 if neatQA.DO_BIDIRECTIONAL_MEMORY then
      print("16101")
      print(paramdxR:norm())
 end

      auxiliary.clipGradients(paramdxA)
      auxiliary.clipGradients(paramdxR)
      auxiliary.updateParametersWithMomentum(paramxA,paramdxA)
      auxiliary.updateParametersWithMomentum(paramxR,paramdxR)

if neatQA.DO_BIDIRECTIONAL_MEMORY then
   backwards_network:bp(attended_input_tensors,neatQA.maximalLengthOccurringInInput[1], actorGradient)
end


  end
 















 
  assert(not(train_attention_network))
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

-- cutorch.synchronize() 

--  auxiliary.printMemory("612")
--assert(false)
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
               if  (true and math.random() < 0.001) then
if false then
                  auxiliary.deepPrint(neatQA.inputTensors, function (tens) return tens[l] end)
end

                  for j=2,neatQA.maximalLengthOccurringInInput[1] do
                    if neatQA.inputTensors[j][l] == 0 then
                       break
                    end

                  --local predictedScoresLM, predictedTokensLM = torch.min(reader_output[j-1][l],1)
auxiliary.write((readDict.chars[neatQA.inputTensors[j][l]]))
if neatQA.ALSO_DO_LANGUAGE_MODELING then
         local predictedScoresLM, predictedTokensLM = torch.min(reader_output[j-1][l],1)
  auxiliary.write(readDict.chars[predictedTokensLM[1]])
  auxiliary.write(math.exp(-predictedScoresLM[1]))
  auxiliary.write(math.exp(-reader_output[j-1][l][neatQA.inputTensors[j][l]]))
end
auxiliary.write(attention_decisions[j][l][1])
auxiliary.write(attention_scores[j][l][1])
io.write("\n")

                  end


               end
--               print(readChunks.corpus[l].text)
               print("ANSW       "..answerID)
               print("PROB       "..actor_output[l][answerID])
               local predictedScore,predictedAnswer = torch.max(actor_output[l],1)
--               print(predictedAnswer)
  --             print(predictedScore)
               print("PREDICTED  "..predictedAnswer[1].." # "..predictedScore[1])
--               local negSample = math.random(math.min(10, actor_output[l]:size()[1]))
  --             print("NEGATIVE EX PROB "..actor_output[l][negSample].." ("..negSample..")")
    --           if (math.abs(actor_output[l][answerID]) <= math.abs(actor_output[l][negSample])) then
               if (answerID == predictedAnswer[1]) then
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




