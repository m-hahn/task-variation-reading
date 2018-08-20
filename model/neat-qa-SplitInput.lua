-- neat-qa-SplitInput.lua was split from neat-qa-Unrolled-Attention.lua (11/25/2016)
-- This is for soft attention. Unrolled-Attention is for hard attention of the vanilla LSTM model, and also has soft atention functionality.
-- UNDO-CHANGES is for vanilla LSTM without hard attention.

neatQA = {}
neatQA.__name = "neat-qa-SplitInput.lua"

neatQA.number_of_LSTM_layers = 1

neatQA.ALSO_DO_LANGUAGE_MODELING = false

neatQA.ACCESS_MEMORY = true

assert(neatQA.ACCESS_MEMORY)

neatQA.INITIALIZE_FROM_NEAT = false--true--false

neatQA.ATTENTION_DOES_Q_ATTENTION = true
assert(neatQA.ATTENTION_DOES_Q_ATTENTION)

neatQA.DO_BIDIRECTIONAL_MEMORY = true--true--false
assert(neatQA.DO_BIDIRECTIONAL_MEMORY)


neatQA.USE_ATTENTION_NETWORK = true
assert(neatQA.USE_ATTENTION_NETWORK)
neatQA.USE_PRETRAINED_EMBEDDINGS = true
assert(neatQA.USE_PRETRAINED_EMBEDDINGS)
neatQA.GET_MORE_THAN_EMBEDDINGS_FROM_NEAT = false
assert(not(neatQA.GET_MORE_THAN_EMBEDDINGS_FROM_NEAT))
---------------------

assert(params.lr == 0)

-- things that can vary in the Sandbox setting
neatQA.CUT_OFF_BELOW_BASELINE = true
if neatQA.CUT_OFF_BELOW_BASELINE then
   neatQA.reluLayer = nn.ReLU():cuda()
end
if false then
 neatQA.squareLayer = nn.Square():cuda()
end


neatQA.NEW_COMBINATION = true
assert(neatQA.NEW_COMBINATION)
-- false+true is BILINEAR
-- true+true is BIAFFINE
-- false+false is 51,53
neatQA.BIAFFINE_ATTENTION = false --false --true
neatQA.USE_BILINEAR_TERM_IN_HARD_ATTENTION = true --false --true--false
assert(not(neatQA.BIAFFINE_ATTENTION) or neatQA.USE_BILINEAR_TERM_IN_HARD_ATTENTION)
neatQA.STORE_Q_ATTENTION = true

neatQA.better_logsigmoid_gradients = true
neatQA.ATTENTION_EMBEDDINGS_FROM_READER = false
neatQA.DO_TRAINING_ON_ATTENTION_EMBEDDINGS = false
neatQA.APPLY_MLP_TO_ATTENTION_EMBEDDING = false
neatQA.STRETCH_Q_EMBEDDINGS = true
assert(not(neatQA.APPLY_MLP_TO_ATTENTION_EMBEDDING))
assert(not(neatQA.DO_TRAINING_ON_ATTENTION_EMBEDDINGS))
assert(not(neatQA.APPLY_MLP_TO_ATTENTION_EMBEDDING) or neatQA.USE_BILINEAR_TERM_IN_HARD_ATTENTION)
neatQA.APPLY_L2_REGULARIZATION_TO_ATT = true
params.l2_regularization = 0.00001

neatQA.rewardBasedOnLogLikeLoss = true --false --true --false --true --false--true --false --false
--assert(neatQA.rewardBasedOnLogLikeLoss)

neatQA.ANALYTICAL_MINIMIZATION_OF_FIXATIONS = true
print("Minimizing fixations analytically?"..tostring(neatQA.ANALYTICAL_MINIMIZATION_OF_FIXATIONS))
neatQA.USE_ADAM_FOR_REINFORCE = true
print("Using Adam? "..tostring(neatQA.USE_ADAM_FOR_REINFORCE))
neatQA.USE_GOLD_LIKELIHOODS_AS_BASELINE = false
assert(not(neatQA.USE_GOLD_LIKELIHOODS_AS_BASELINE))
neatQA.CENTER_PREDICTORS = true
---------------------

neatQA.scaleAttentionWithLength = true
assert(neatQA.scaleAttentionWithLength)

neatQA.USE_INNOVATIVE_ATTENTION = false
assert(not(neatQA.USE_INNOVATIVE_ATTENTION))

neatQA.useBackwardQForAttention = false
print("Using backward Q for attention?")
print(neatQA.useBackwardQForAttention)
assert(not(neatQA.useBackwardQForAttention))



neatQA.use_l1 = true
print("Use L1 for fixation rate?   "..tostring(neatQA.use_l1))


neatQA.PRETRAINED_ATTENTION = false --true
assert(not(neatQA.PRETRAINED_ATTENTION))
assert(not(neatQA.PRETRAINED_ATTENTION) or neatQA.USE_INNOVATIVE_ATTENTION)

assert(not(neatQA.GET_MORE_THAN_EMBEDDINGS_FROM_NEAT))

QUESTION_LENGTH = 30 --50 --50 --50 --50

assert(neatQA.ACCESS_MEMORY)
assert(not(neatQA.GET_MORE_THAN_EMBEDDINGS_FROM_NEAT and (not neatQA.INITIALIZE_FROM_NEAT)))

assert( not (neatQA.USE_PRETRAINED_EMBEDDINGS and neatQA.INITIALIZE_FROM_NEAT))

print("neatQA")-- (if CONDITION is not present, it's 'preview')")
print(neatQA)



require('auxiliary')
require('nn.RecursorMod')
require('nn.SequencerMod')
require('nn.PrintLayer')
require('nn.BlockGradientLayer')


require('qaAttentionAnswerer')
require('recurrentNetworkOnSequence')

require('qaReinforce')

function neatQA.createAnswerNetwork(param,gradparam)
    return qaAttentionAnswerer.createAnswerNetworkWithMemoryAndQuestionAttention(param,gradparam)
end


function neatQA.setup()
  print("Creating a RNN LSTM network.")

-- ######################
neatQA.CONDITION = "preview" -- "preview"
if CONDITION_FROM_SETPARAMETERS ~= nil then
   neatQA.CONDITION = CONDITION_FROM_SETPARAMETERS
end
assert(neatQA.CONDITION == "preview" or neatQA.CONDITION == "nopreview" or neatQA.CONDITION == "mixed" or neatQA.CONDITION == "fullpreview" or neatQA.CONDITION == "fullnopreview")
print("CONDITION "..neatQA.CONDITION)
params.CONDITION = neatQA.CONDITION

params.useBackwardQForAttention = neatQA.useBackwardQForAttention


  print("Setting params.init_weight (60378)")
  params.init_weight=0.01


  -- initialize data structures
  model.dsR = {}
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
     attention_decisions[i] = torch.CudaTensor(params.batch_size,1)
     attention_scores[i] = torch.CudaTensor(params.batch_size,1)
     baseline_scores[i] = torch.CudaTensor(params.batch_size,1)
     attended_input_tensors[i] = torch.CudaTensor(params.batch_size,1)
     attention_probabilities[i] = torch.CudaTensor(params.batch_size,1)
  end

  probabilityOfChoices = torch.FloatTensor(params.batch_size)
  totalAttentions = torch.FloatTensor(params.batch_size) -- apparently using CudaTensor would cause a noticeable slowdown...?!
  nll = torch.FloatTensor(params.batch_size)

  attention_inputTensors = {}


  ones = transfer_data(torch.ones(params.batch_size))
  rewardBaseline = 0


  local embeddings = nil
  local embeddingsGrad = nil
  if neatQA.USE_PRETRAINED_EMBEDDINGS then
--        local parameters, _ = reader_core_network:parameters()
        embeddings = torch.CudaTensor(params.vocab_size+1,params.embeddings_dimensionality)
        embeddingsGrad = torch.CudaTensor(params.vocab_size+1,params.embeddings_dimensionality):zero()
        readDict.setToPretrainedEmbeddings(embeddings)
    if true then
     print(embeddings[readDict.word2Num("beer")+1]*embeddings[readDict.word2Num("wine")+1])
     print(embeddings[readDict.word2Num("computer")+1]*embeddings[readDict.word2Num("wine")+1])
     print(embeddings[readDict.word2Num("paper")+1]*embeddings[readDict.word2Num("wine")+1])
     print(embeddings[readDict.word2Num("drink")+1]*embeddings[readDict.word2Num("wine")+1])
     --print(embeddings[readDict.word2Num("towel")+1]*embeddings[readDict.word2Num("paper")+1])
     print(embeddings[readDict.word2Num("paper")+1]*embeddings[readDict.word2Num("article")+1])
     print(embeddings[readDict.word2Num("paper")+1]*embeddings[readDict.word2Num("stone")+1])
     print(embeddings[readDict.word2Num("paper")+1]*embeddings[readDict.word2Num("elephant")+1])
     print(embeddings[readDict.word2Num("paper")+1]*embeddings[readDict.word2Num("bicycle")+1])
    end
  end

local params2, sentencesRead ,SparamxForward, SparamdxForward, SparamxBackward, SparamdxBackward, SparamxQForward, SparamdxQForward, SparamxQBackward, SparamdxQBackward, SparamxA, SparamdxA, SparamxRA, SparamdxRA
if LOAD then

     print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)
     local storedModel = torch.load(BASE_DIRECTORY.."/model-"..fileToBeLoaded, "binary")
--     params2, sentencesRead ,SparamxForward, SparamdxForward, SparamxBackward, SparamdxBackward, SparamxQForward, SparamdxQForward, SparamxQBackward, SparamdxQBackward, SparamxA, SparamdxA, SparamxRA, SparamdxRA = unpack(storedModel)
print("Parameters from the stored Model:")
print(storedModel)

params2 = storedModel.params
sentencesRead       = storedModel.readWords
SparamxForward      = storedModel.SparamxForward
SparamdxForward      = storedModel.SparamdxForward
SparamxBackward      = storedModel.SparamxBackward
SparamdxBackward      = storedModel.SparamdxBackward
SparamxQForward      = storedModel.SparamxQForward
SparamdxQForward      = storedModel.SparamdxQForward
SparamxQBackward      = storedModel.SparamxQBackward
SparamdxQBackward      = storedModel.SparamdxQBackward
SparamxA      = storedModel.SparamxA
SparamdxA      = storedModel.SparamdxA
SparamxRA      = storedModel.SparamxRA
SparamdxRA      = storedModel.SparamdxRA

if params2.CONDITION ~= nil then
  if params2.CONDITION ~= params.CONDITION then
     print("Warning: condition is different from previous run.")
  end
end


--           modelsArray = {params,(numberOfWords/params.seq_length),SparamxForward, SparamdxForward, SparamxBackward, SparamdxBackward, SparamxQForward, SparamdxQForward, SparamxQBackward, SparamdxQBackward, SparamxA, SparamdxA, SparamxRA, SparamdxRA}
  embeddingsShrinkingFactor = 10.0
  if #SparamxRA == 0 then
    print("Pretrained embeddings for attention. "..tostring(neatQA.ATTENTION_EMBEDDINGS_FROM_READER))
    if neatQA.ATTENTION_EMBEDDINGS_FROM_READER then

    embeddingsForAtt = SparamxForward[1]
    embeddingsGradForAtt = SparamdxForward[1]:clone()
    embeddingsGradForAtt:zero()
    -- normalize them
    -- adapted from torch.renorm
      embeddingsForAtt = embeddingsForAtt:clone()
      -- collapse non-dim dimensions:
      local norms = embeddingsForAtt:norm(2,2):mul(embeddingsShrinkingFactor)
      -- renormalize
      embeddingsForAtt:cdiv(norms:expandAs(embeddingsForAtt))


    else -- note that randomly initialized elements have to be renormed first
      embeddingsForAtt = embeddings:clone()
      local norms = embeddingsForAtt:norm(2,2):mul(embeddingsShrinkingFactor)
      -- renormalize
      embeddingsForAtt:cdiv(norms:expandAs(embeddingsForAtt))



--       embeddingsForAtt = torch.div(embeddings,embeddingsShrinkingFactor)
       embeddingsGradForAtt = embeddingsGrad
    end

     


    if true then
     print("Some similarities:")
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("beer")+1]*embeddingsForAtt[readDict.word2Num("beer")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num(".")+1]*embeddingsForAtt[readDict.word2Num(".")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("beer")+1]*embeddingsForAtt[readDict.word2Num("wine")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("computer")+1]*embeddingsForAtt[readDict.word2Num("wine")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("paper")+1]*embeddingsForAtt[readDict.word2Num("wine")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("drink")+1]*embeddingsForAtt[readDict.word2Num("wine")+1])
     --print(embeddingsForAtt[readDict.word2Num("towel")+1]*embeddingsForAtt[readDict.word2Num("paper")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("paper")+1]*embeddingsForAtt[readDict.word2Num("article")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("paper")+1]*embeddingsForAtt[readDict.word2Num("stone")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("paper")+1]*embeddingsForAtt[readDict.word2Num("elephant")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("paper")+1]*embeddingsForAtt[readDict.word2Num("bicycle")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("@entity5")+1]*embeddingsForAtt[readDict.word2Num("@entity7")+1])
     print(embeddingsShrinkingFactor*embeddingsShrinkingFactor*embeddingsForAtt[readDict.word2Num("@entity5")+1]*embeddingsForAtt[readDict.word2Num("@entity5")+1])

    end



    SparamxRA      = {embeddingsForAtt} --embeddingsFromReader}
    SparamdxRA    = {embeddingsGradForAtt} --embeddingsGradFromReader}
  end

    print(params2)
  if params.useBackwardQForAttention then
    assert(false)
    assert(params2.useBackwardQForAttention == false or #SparamxRA == 1)
  end
--crash()
else
  SparamxForward = {embeddings}
  SparamdxForward = {embeddingsGrad}
  SparamxBackward = {embeddings}
  SparamdxBackward = {embeddingsGrad}
  SparamxQForward = {embeddings}
  SparamdxQForward = {embeddingsGrad}
  SparamxQBackward = {embeddings}
  SparamdxQBackward = {embeddingsGrad}
  SparamxA = {}
  SparamdxA = {}
  SparamxRA      = {embeddings}
  SparamdxRA    = {embeddingsGrad} 
end


if (not IS_CONTINUING_ATTENTION) then
  assert(not(DOING_EVALUATION_OUTPUT))
  print("WARNING: CREATING FRESH ATTENTION NETWORK")
  SparamxRA = {embeddings}
  SparamdxRA = {embeddingsGrad}
end

if false and neatQA.CONDITION ~= "mixed" then
 setupBidirBaseline(SparamxForward, SparamxB, SparamdxB)
else
-- assert(neatQA.CONDITION == "mixed")
 setupSimpleBaseline(SparamxB, SparamdxB)
end




     forward_network = RecurrentNetworkOnSequence.new(params.rnn_size,SparamxForward,SparamdxForward,params.seq_length)
     backward_network = RecurrentNetworkOnSequence.new(params.rnn_size, SparamxBackward,SparamdxBackward,params.seq_length)
     question_forward_network = RecurrentNetworkOnSequence.new(params.rnn_size, SparamxQForward,SparamdxQForward, QUESTION_LENGTH)
     question_backward_network =  RecurrentNetworkOnSequence.new(params.rnn_size, SparamxQBackward,SparamdxQBackward, QUESTION_LENGTH)



   -- II execute getParameters()
   actor_core_network = neatQA.createAnswerNetwork(SparamxA, SparamdxA)
   paramxA, paramdxA = actor_core_network:getParameters()
   paramdxA:zero()


assert(neatQA.USE_ATTENTION_NETWORK)
   -- III build clones
   if neatQA.USE_ATTENTION_NETWORK then
      -- TODO put in parameters
      attentionNetwork = attention.createAttentionBilinear(SparamxRA,SparamdxRA) 
      if false and neatQA.CONDITION == "preview" and neatQA.PRETRAINED_ATTENTION and #(SparamxRA) == 0 then
        assert(false)
        print("Fetching attention map from answer module.")
        assert(neatQA.USE_INNOVATIVE_ATTENTION)
        assert(neatQA.CONDITION == "preview")
        assert(#(SparamxRA) == 0)
        local pRA
        local pdRA
        pRA, pdRA = attentionNetwork:parameters()
        local bilinearTarget = pRA[4]
        local pA
        local pdA
        pA, pdA = actor_core_network:parameters()
        local bilinearSource = pA[1]
        bilinearTarget = bilinearTarget:narrow(3,129,128)
        bilinearSource = bilinearSource:narrow(2,1,128)
        bilinearSource = bilinearSource:transpose(2,3)
        print(bilinearTarget:size())
        print(bilinearSource:size())
        bilinearTarget:copy(bilinearSource)
      elseif neatQA.PRETRAIND_ATTENTION then
        print("Not taking existing attention.")
      end
      paramxRA, paramdxRA = attentionNetwork:getParameters()
      paramdxRA:zero()
      attentionNetworks = {}
      auxiliary.buildClones(params.seq_length,attentionNetworks,attentionNetwork)
   end


   vectorOfLengths = torch.LongTensor(params.batch_size)
   neatQA.maximalLengthOccurringInInput = {0}
   neatQA.maximalLengthOccurringInInputQuestion = {0}



if true or neatQA.CONDITION == "preview" then
   local y0 = nn.Identity()()
   -- y0 consists of the one-hot vectors of the questions
   questionTokensReshaped =  nn.JoinTable(1)(y0)
   questionTokensEmbeddings = nn.LookupTableMaskZero(params.vocab_size,params.embeddings_dimensionality)(questionTokensReshaped)
   questionTokensEmbeddings = nn.View(-1,params.batch_size,params.embeddings_dimensionality)(questionTokensEmbeddings)
   questionTokensEmbeddings = nn.Transpose({1,2})(questionTokensEmbeddings)
   neatQA.questionEncoder = nn.gModule({y0},{questionTokensEmbeddings}):cuda()
   questionEncParams, questionEncGradparams = neatQA.questionEncoder:parameters()
   assert(#(questionEncParams) == 1)
   paramsAtt, paramdsAtt = attentionNetwork:parameters()
  -- local indexTable = {}
  -- indexTable[false] = 1
--   indexTable[true] = 3
   embeddingsIndex = 1 --indexTable[neatQA.BIAFFINE_ATTENTION]
   questionEncParams[1]:copy(paramsAtt[embeddingsIndex])
   questionEncParams[1][1]:zero() -- the padding should not influence attention
   questionEncGradparams[1]:zero()
   if neatQA.STRETCH_Q_EMBEDDINGS then
      questionEncParams[1]:mul(embeddingsShrinkingFactor*embeddingsShrinkingFactor)
   end
end


neatQA.questionHistory = {}
neatQA.questionHistory[0] = torch.CudaTensor(params.batch_size,1):zero()
neatQA.questionHistoryFromSource = {}
neatQA.questionHistoryFromSource[0] = torch.CudaTensor(params.batch_size,1):zero()

neatQA.historyGradientStart = torch.CudaTensor(params.batch_size, 1):zero()
neatQA.historyFromSourceGradientStart = torch.CudaTensor(params.batch_size, 1):zero()




neatQA.positionTensors = {}
for i=1, params.seq_length do
  table.insert(neatQA.positionTensors, torch.CudaTensor(params.batch_size,1):fill(i/(params.seq_length+0.0)))
end

--[[if neatQA.CONDITION == "mixed" then
       neatQA.condition_mask = torch.CudaTensor(params.batch_size) 
       print(neatQA.condition_mask)
end]]


end


function neatQA.parameters()
   local parameters = {}
   local gradParameters = {}
   local modules = {forward_network, backward_network, question_forward_network, question_backward_network}
   for q=1, #modules do
     local p, dp = modules[1]:parameters()
     table.insert(parameters,p)
     table.insert(gradParameters,dp)
   end
   return parameters, gradParameters
end




function neatQA.fp(corpus, startIndex, endIndex)

if neatQA.USE_GOLD_LIKELIHOODS_AS_BASELINE then
   assert(false)
   neatQA.baselineTensor = torch.FloatTensor(params.batch_size):zero()
   for i=startIndex, endIndex do
     neatQA.baselineTensor[i-startIndex+1] = corpus[i].baseline
   end
   neatQA.baselineTensor = neatQA.baselineTensor:cuda()
end

   neatQA.maxLengthsPerItem = torch.LongTensor(params.batch_size)

    neatQA.inputTensors, neatQA.inputTensorsQuestion = auxiliary.buildSeparateInputTensorsQA(corpus,startIndex,endIndex,neatQA.maxLengthsPerItem,neatQA.maximalLengthOccurringInInput, neatQA.maximalLengthOccurringInInputQuestion)

  neatQA.answerTensors =  qa.buildAnswerTensor(corpus, startIndex, endIndex)

--print("4691052")
--print(neatQA.maxLengthsPerItem)

-----------------------
-----------------------
-- Read the Question --
-----------------------
-----------------------


  question_forward_cs, question_forward_hs = question_forward_network:fp(neatQA.inputTensorsQuestion,neatQA.maximalLengthOccurringInInputQuestion[1], nil)
  question_backward_cs, question_backward_hs = question_backward_network:fp(auxiliary.reverseTable(neatQA.inputTensorsQuestion,neatQA.maximalLengthOccurringInInputQuestion[1]),neatQA.maximalLengthOccurringInInputQuestion[1], nil)


-----------------------
-----------------------
---- Read the Text ----
-----------------------
-----------------------

attended_input_tensors = auxiliary.shallowCopyTable(neatQA.inputTensors)


  for i=1, neatQA.maximalLengthOccurringInInput[1] do
if not neatQA.USE_ATTENTION_NETWORK then
assert(false)
         attention_decisions[i] = attention_decisions[i]:view(-1)
         attended_input_tensors[i], _ = hardAttention.makeAttentionDecisions(i, neatQA.inputTensors[i])
         attention_decisions[i] = attention_decisions[i]:view(params.batch_size,1)
end
  end


  print("40  "..neatQA.maximalLengthOccurringInInput[1])

attentionObjects = {attentionNetworks = attentionNetworks,questionForward = question_forward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], questionBackward=question_backward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], decisions=attention_decisions,scores=attention_scores, originalInputTensors = neatQA.inputTensors, probabilities=attention_probabilities, questionInputTensors=neatQA.inputTensorsQuestion}



  -- Reads the text, calling the attention module (and writing results in attended_input_tensors and attentionObjects) for each word
  forward_cs, forward_hs = forward_network:fp(attended_input_tensors,neatQA.maximalLengthOccurringInInput[1], attentionObjects)


  -- Reads the text backward (respecting the choices made by the hard attention)
  backward_cs, backward_hs = backward_network:fp(auxiliary.reverseTable(attended_input_tensors, neatQA.maximalLengthOccurringInInput[1]),neatQA.maximalLengthOccurringInInput[1], nil)

----------------------
----------------------
--- Answer Softmax ---
----------------------
----------------------

  neatQA.actorInput ={forward_hs, auxiliary.reverseTable(backward_hs, neatQA.maximalLengthOccurringInInput[1]), question_forward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], question_backward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]]}

  actor_output = actor_core_network:forward(neatQA.actorInput):float()
---------------------------------
----------------------------------

  for i=1, params.batch_size do
    nll[i] = - actor_output[i][neatQA.answerTensors[i]]
  end

  meanNLL = 0.95 * meanNLL + 0.05 * nll:mean()

  return nll, actor_output
end


function neatQA.bp(corpus, startIndex, endIndex)
  auxiliary.prepareMomentum(paramdxA)

if false then
  reset_ds()
end

if false then  
  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true
end

  if (params.lr_att > 0 or params.lr > 0) and (true or train_autoencoding) then --hrhr
------------ CRITERION
   if params.lr > 0 then
      derivativeFromCriterion = neatQA.criterionDerivative
      derivativeFromCriterion:zero()
      for i=1, params.batch_size do
        derivativeFromCriterion[i][neatQA.answerTensors[i]] = -1
      end
    end
   assert(neatQA.DO_BIDIRECTIONAL_MEMORY)

-----------------------------------------------
------------ ACTOR ----------------------------
-----------------------------------------------
if not true then
 print("Setting to zero 26825")
 paramdxA:zero()
end



    local actorGradient 
   if params.lr > 0 then
        actorGradient = actor_core_network:backward(neatQA.actorInput, transfer_data(derivativeFromCriterion))

        auxiliary.clipGradients(paramdxA)
        auxiliary.updateParametersWithMomentum(paramxA,paramdxA)
   else
      actorGradient = {}
   end

--------------------------------------------------------
----------------- BACKWARD PASS FOR READERS ------------
--------------------------------------------------------


attentionObjects = {attentionNetworks = attentionNetworks,questionForward = question_forward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], questionBackward=question_backward_hs[neatQA.maximalLengthOccurringInInputQuestion[1]], decisions=attention_decisions,scores=attention_scores, originalInputTensors = neatQA.inputTensors, probabilities = attention_probabilities,questionInputTensors=neatQA.inputTensorsQuestion}


assert(neatQA.USE_ATTENTION_NETWORK)


   attentionGradients = forward_network:bp(attended_input_tensors,neatQA.maximalLengthOccurringInInput[1], {nil, nil,nil,actorGradient[1]}, attentionObjects)
   if params.lr > 0 then
     backward_network:bp(auxiliary.reverseTable(attended_input_tensors, neatQA.maximalLengthOccurringInInput[1]),neatQA.maximalLengthOccurringInInput[1], {nil, nil, nil, auxiliary.reverseTable(actorGradient[2], neatQA.maximalLengthOccurringInInput[1])},nil)
     question_forward_network:bp(neatQA.inputTensorsQuestion,neatQA.maximalLengthOccurringInInputQuestion[1], {nil,actorGradient[3],nil,nil},nil)
     question_backward_network:bp(auxiliary.reverseTable(neatQA.inputTensorsQuestion,   neatQA.maximalLengthOccurringInInputQuestion[1] ),neatQA.maximalLengthOccurringInInputQuestion[1], {nil,actorGradient[4],nil,nil},nil)
   end


  end
--  neatQA.doBackwardForAttention()
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
              
               local answerID = qa.getFromAnswer(readChunks.corpus,l,1)
               if answerID == nil then
                    print("463: answerID == nil")
                    answerID = 1
               end
               if  (DOING_DEBUGGING or (CREATE_RECORDED_FILES and DOING_EVALUATION_OUTPUT) or false or math.random() < 0.0001) then
--                  auxiliary.deepPrint(neatQA.inputTensors, function (tens) return tens[l] end)
                  print(45825)
print("QUESTION:")
for j=1,QUESTION_LENGTH do
   local tokenID = neatQA.inputTensorsQuestion[j][l]
   if tokenID == 0 then
     break
   end
   io.write(readDict.chars[tokenID].."  ")
end
print("\nTEXT:")
                  for j=1,neatQA.maximalLengthOccurringInInput[1] do
if false then
print("...")
print(j)
print(l)
print(neatQA.inputTensors[j])
print(neatQA.inputTensors[j][l])
end

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
auxiliary.write(attention_decisions[j][l][1])
auxiliary.write(attention_scores[j][l][1])
if neatQA.ACCESS_MEMORY then
   auxiliary.write(globalForExpOutput.softAttentionsContainer.output[l][1][j])
end
io.write("\n")


                  end


               end
               print("ANSW       "..answerID)
               print("PROB       "..actor_output[l][answerID])
               local predictedScore,predictedAnswer = torch.max(actor_output[l],1)
               print("PREDICTED  "..predictedAnswer[1].." # "..predictedScore[1])
if neatQA.USE_GOLD_LIKELIHOODS_AS_BASELINE then
 assert(false)
               print("GOLD       "..neatQA.baselineTensor[l])
end
               if (answerID == predictedAnswer[1]) then
                 correct = correct + 1.0
               else
                 incorrect = incorrect + 1.0
               end
            end
            print("APPROX PERFORMANCE  "..(correct / (correct + incorrect)))
            globalForExpOutput.accuracy = 0.95 * globalForExpOutput.accuracy + 0.05 * (correct / (correct+incorrect))

            print("Avg performance     "..(globalForExpOutput.accuracy))
end




