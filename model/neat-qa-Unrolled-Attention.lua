-- neat-qa-Unrolled-Attention.lua was split from neat-qa-UNDO-THE-CHANGES.lua (11/16/2016)
assert(false)

neatQA = {}
neatQA.__name = "neat-qa-Unrolled-Attention.lua"

neatQA.number_of_LSTM_layers = 1

neatQA.ALSO_DO_LANGUAGE_MODELING = false

neatQA.ACCESS_MEMORY = false--true

neatQA.INITIALIZE_FROM_NEAT = true
neatQA.DO_BIDIRECTIONAL_MEMORY = false --true--true--false
neatQA.rewardBasedOnLogLikeLoss = false--true --false
neatQA.USE_ATTENTION_NETWORK = true--false


print(neatQA)



require('auxiliary')
require('nn.RecursorMod')
require('nn.SequencerMod')
require('nn.PrintLayer')
require('nn.BlockGradientLayer')


require('qaAttentionAnswerer')
require('recurrentNetworkOnSequence')

require('qaReinforce')

-- input: recurrent state and memory cells
-- output: softmax layer
function neatQA.createSimpleAnswerNetwork()
  local model = nn.Sequential()
  model:add(nn.JoinTable(2))
  model:add(nn.Linear(2*params.rnn_size,NUMBER_OF_ANSWER_OPTIONS))
  model:add(nn.LogSoftMax())
--  assert(false, "This should presumably become a GPU network")

  return model:cuda()
end

function neatQA.createAnswerNetwork()
  if neatQA.ACCESS_MEMORY then
    return qaAttentionAnswerer.createAnswerNetworkWithMemoryAttention()
  else
    return neatQA.createSimpleAnswerNetwork()
  end
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
  
  assert(neatQA.number_of_LSTM_layers == 1)

    reader_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero() 
    reader_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()



  neatQA.criterionDerivative = torch.DoubleTensor(params.batch_size, NUMBER_OF_ANSWER_OPTIONS) 


  attention_decisions = {}
  attention_scores = {}
  baseline_scores = {}
  attended_input_tensors = {}
  attention_probabilities = {}
  for i=1, params.seq_length do
     attention_decisions[i] = torch.CudaTensor(params.batch_size)
     attention_scores[i] = torch.CudaTensor(params.batch_size,1)
     baseline_scores[i] = torch.CudaTensor(params.batch_size,1)
     attended_input_tensors[i] = torch.CudaTensor(params.batch_size,1)
     attention_probabilities[i] = torch.CudaTensor(params.batch_size,1)
  end

  probabilityOfChoices = torch.FloatTensor(params.batch_size)
  totalAttentions = torch.FloatTensor(params.batch_size) -- apparently using CudaTensor would cause a noticeable slowdown...?!
  nll = torch.FloatTensor(params.batch_size)

  attention_inputTensors = {}


  ones = torch.ones(params.batch_size):cuda()
  rewardBaseline = 0


  local reader_core_network
  reader_core_network = autoencoding.create_network(neatQA.ALSO_DO_LANGUAGE_MODELING, true, true)
  actor_core_network = neatQA.createAnswerNetwork()
  attentionNetwork = attention.createAttentionNetwork() --createAttentionNetwork()

  if ((not LOAD) and neatQA.INITIALIZE_FROM_NEAT) then

      print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)
     
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

  elseif LOAD then

     print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, SparamxRA, SparamdxRA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load(BASE_DIRECTORY.."/model-"..fileToBeLoaded, "binary"))

     if SparamxB == nil and USE_BIDIR_BASELINE and DO_TRAINING and IS_CONTINUING_ATTENTION then
        print("962 no baseline in saved file")
        crash()
     end

     print(params2)

    -- LOAD PARAMETERS
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()
     if (#reader_network_params ~= #SparamxR) then
       print("WARNING")
       print(SparamdxR)
       print(reader_network_params)
     end


     for j=1, #reader_network_params do
           reader_network_params[j]:set(SparamxR[j])
           reader_network_gradparams[j]:set(SparamxR[j])
     end
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()

------
     -- ACTOR
     
     actor_network_params, actor_network_gradparams = actor_core_network:parameters()
     assert(#SparamxA == #actor_network_params)
     for j=1, #SparamxA do
           actor_network_params[j]:set(SparamxA[j]:cuda())
           actor_network_gradparams[j]:set(SparamdxA[j]:cuda())
     end
     

     -- ATTENTION
     att_network_params, network_gradparams = attentionNetwork:parameters()
     print("Attention")
     print(att_network_params)
     print("Reader")
     print(reader_network_params)

     if params.ATTENTION_WITH_EMBEDDINGS then
        if not IS_CONTINUING_ATTENTION then
           assert(att_network_params[1]:size(1) == params.vocab_size+1)
           assert(att_network_params[1]:size(2) == params.embeddings_dimensionality)
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
     question_network = RecurrentNetworkOnSequence.new(params.rnn_size, reader_core_network:parameters())
   end


   -- II execute getParameters()
   paramxR, paramdxR = reader_core_network:getParameters()
   paramxA, paramdxA = actor_core_network:getParameters()
   paramxRA, paramdxRA = attentionNetwork:getParameters()

   print("Building clones")
   -- III build clones
   readerRNNs = {}
   attentionNetworks = {}
   auxiliary.buildClones(params.seq_length,readerRNNs,reader_core_network)
   auxiliary.buildClones(params.seq_length,attentionNetworks,attentionNetwork)

     -- IV zero derivatives
   paramdxRA:zero()
   paramdxA:zero()
   paramdxR:zero()

   -- initialize more bookkeeping
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
  neatQA.maxEndOfQuestions = -1

  -- since we want the length to be bounded by the input for attention, remove all the later entries
  neatQA.emptyCHTables()

  probabilityOfChoices:fill(1)
--  totalAttentions:fill(params.ATTENTION_VALUES_BASELINE)
  neatQA.inputTensors = auxiliary.buildInputTensorsQA(corpus, startIndex, endIndex, vectorOfLengths, neatQA.maximalLengthOccurringInInput)
  neatQA.inputTensorsTables = auxiliary.toUnaryTables(neatQA.inputTensors)

  neatQA.answerTensors =  qa.buildAnswerTensor(corpus, startIndex, endIndex)



  
  print("40  "..neatQA.maximalLengthOccurringInInput[1])
  for i=1, neatQA.maximalLengthOccurringInInput[1] do
     if neatQA.USE_ATTENTION_NETWORK then
        attention_scores[i], attention_decisions[i], attention_probabilities[i],  attended_input_tensors[i]  = unpack(attentionNetworks[i]:forward({neatQA.inputTensors[i], reader_h[i-1], reader_c[math.min(i-1,neatQA.maxEndOfQuestions)]}))
     else
         local inputTensor = neatQA.inputTensors[i]
         attention_decisions[i] = attention_decisions[i]:view(-1)
         attended_input_tensors[i], _ = hardAttention.makeAttentionDecisions(i, inputTensor)
         attention_decisions[i] = attention_decisions[i]:view(params.batch_size,1)
     end
     reader_c[i], reader_h[i], reader_output[i] = unpack(readerRNNs[i]:forward({attended_input_tensors[i], reader_c[i-1], reader_h[i-1]}))
  end
  neatQA.readerCFinal = reader_c[neatQA.maximalLengthOccurringInInput[1]]
  neatQA.readerHFinal = reader_h[neatQA.maximalLengthOccurringInInput[1]]
  if neatQA.ACCESS_MEMORY then
     actor_output = actor_core_network:forward({neatQA.readerCFinal, neatQA.readerHFinal, reader_c}):float()
  else
     actor_output = actor_core_network:forward({neatQA.readerCFinal, neatQA.readerHFinal}):float()
  end
  for i=1, params.batch_size do
    nll[i] = - actor_output[i][neatQA.answerTensors[i]]
  end

if neatQA.DO_BIDIRECTIONAL_MEMORY then
  local cs, hs = backwards_network:fp(attended_input_tensors,neatQA.maximalLengthOccurringInInput[1])
  local cs1, hs1 = question_network:fp(attended_input_tensors,neatQA.maximalLengthOccurringInInput[1])

  if true then
   for u =1,50 do
--     local i = math.random(60)
  --   local j = math.random(100)
     print("...")
     print(cs[5][u][u])
     print(reader_c[5][u][u])
     print(cs1[5][u][u])
   end
  end
end



  meanNLL = 0.95 * meanNLL + 0.05 * nll:mean()

  return nll, actor_output
end


function neatQA.bp(corpus, startIndex, endIndex)
  auxiliary.prepareMomentum(paramdxR)
  auxiliary.prepareMomentum(paramdxA)

  reset_ds()
  
  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true


  gradientCFromAttention, gradientQFromAttention = neatQA.doBackwardForAttention()

  
  if params.lr > 0 and (true) then --hrhr


 if neatQA.DO_BIDIRECTIONAL_MEMORY then
   print("161012")
   print(paramdxR:norm())
 end




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
      model.dsR[1]:copy(actorGradient[1])
      model.dsR[2]:copy(actorGradient[2])
      for i = neatQA.maximalLengthOccurringInInput[1], 1, -1 do
          if neatQA.ACCESS_MEMORY then 
             model.dsR[1]:add(actorGradient[3][i])
          end
if false then
          if i < neatQA.maxEndOfQuestions then
             model.dsR[1]:add(gradientQFromAttention[i+1])
          elseif i == neatQA.maxEndOfQuestions then
             for j = i+1, neatQA.maximalLengthOccurringInInput[1]-1 do
              model.dsR[1]:add(gradientQFromAttention[j])
            end
          end
end

 if neatQA.DO_BIDIRECTIONAL_MEMORY then
          print("52618 "..i.."  "..model.dsR[1]:norm().."  "..model.dsR[2]:norm())
 end
--print(gradientCFromAttention)
          if i < neatQA.maximalLengthOccurringInInput[1]-1 then
--print(i)
             model.dsR[2]:add(gradientCFromAttention[i+1])
          end

          local tmp = readerRNNs[i]:backward({  attended_input_tensors[i] , reader_c[i-1], reader_h[i-1]},
                                        model.dsR)
          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          cutorch.synchronize()
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
   question_network:bp(attended_input_tensors,neatQA.maximalLengthOccurringInInput[1], actorGradient)
end



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
               if  (false or math.random() < 0.008) then
--                  auxiliary.deepPrint(neatQA.inputTensors, function (tens) return tens[l] end)


                  for j=2,neatQA.maximalLengthOccurringInInput[1] do
                    if neatQA.inputTensors[j][l] == 0 then
                       break
                    end
  --                  local wordProbabilities = neatQA.fullOutput[1][1][j-1][3][l]
--                    local predictedScoresLM, predictedTokensLM = torch.min( wordProbabilities,1)
    --                io.write((readDict.chars[neatQA.inputTensors[j][l]]))--..'\n')
      --              io.write(" \t "..readDict.chars[predictedTokensLM[1]])
        --            io.write("  "..math.exp(-predictedScoresLM[1]))
          --          io.write("  "..math.exp(-wordProbabilities[neatQA.inputTensors[j][l]]))

--                  local predictedScores, predictedTokens = torch.min(actor_output[j][l],1)
--print(attention_decisions[j])
auxiliary.write((readDict.chars[neatQA.inputTensors[j][l]]))
if neatQA.ALSO_DO_LANGUAGE_MODELING then
         local predictedScoresLM, predictedTokensLM = torch.min(reader_output[j-1][l],1)
  auxiliary.write(readDict.chars[predictedTokensLM[1]])
  auxiliary.write(math.exp(-predictedScoresLM[1]))
  auxiliary.write(math.exp(-reader_output[j-1][l][neatQA.inputTensors[j][l]]))
end
auxiliary.write(attention_decisions[j][l][1])
auxiliary.write(attention_scores[j][l][1])
if neatQA.ACCESS_MEMORY then
--print(globalForExpOutput.softAttentionsContainer.output)
--print(j)
--print(l)
--print("66310")
   auxiliary.write(globalForExpOutput.softAttentionsContainer.output[l][1][j])
end
--auxiliary.write(attention_probabilities[j][l][1])
io.write("\n")


--attention_scores[i], attention_decisions[i]
--"\n")
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




