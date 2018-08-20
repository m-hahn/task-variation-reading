attention = {}

require('nn.BernoulliLayer')
require('nn.ReplicateDynamic')
require('nn.BernoulliLayerSigmoidWithLogProb')
require('nn.StoreInContainerLayer')

attention.ABLATE_INPUT = false
attention.ABLATE_STATE = false
attention.ABLATE_SURPRISAL = false


if string.match(params.ablation, 'i') then
  assert(false)
  attention.ABLATE_INPUT = true
end
if string.match(params.ablation, 'r') then
  assert(false)
  attention.ABLATE_STATE = true
end
if string.match(params.ablation, 's') then
  assert(false)
  attention.ABLATE_SURPRISAL = true
end


print("ABLATION INP STATE SURP")
print(attention.ABLATE_INPUT)
print(attention.ABLATE_STATE)
print(attention.ABLATE_SURPRISAL)


   attention.ALLOW_HIDDEN_LAYERS = false
   print("allow hidden layers? at 3535 "..tostring(attention.ALLOW_HIDDEN_LAYERS))


function attention.scalarCoefficient(dim, input, name)
   if (not attention.ALLOW_HIDDEN_LAYERS) then
      return nn.Linear(dim,1)(input)
   else
      print("MLP with hidden layer for "..name.." at 4154")
      local hidden = 50
      return nn.Linear(hidden,1)(nn.ELU()(nn.Linear(dim,hidden)(input)))
   end
end

function attention.createAttentionNewCombinationReduceCollinearity(param,gradparam)

 assert(true or neatQA.CONDITION ~= "nopreview")

 local condition_mask
 if true or neatQA.CONDITION == "mixed" then
   condition_mask = nn.Identity()()
 else
   assert(false)
   --current_condition = neatQA.CONDITION
 end
 --assert(current_condition == "preview" or current_condition == "nopreview")



assert(neatQA.CENTER_PREDICTORS)
 print("Doing biaffine with reduced collinearity at 3754")
assert(neatQA.NEW_COMBINATION)
   assert(neatQA.ATTENTION_DOES_Q_ATTENTION)

   assert(params.TASK == 'neat-qa')


   print("Doing training on attention embeddings?")
   print(neatQA.DO_TRAINING_ON_ATTENTION_EMBEDDINGS)

APPLY_TRAFO_ALSO_TO_Q = false
print("APPLY_TRAFO_ALSO_TO_Q   "..tostring(APPLY_TRAFO_ALSO_TO_Q))
assert(not(neatQA.STRETCH_Q_EMBEDDINGS) or not(APPLY_TRAFO_ALSO_TO_Q))
assert(not(APPLY_TRAFO_ALSO_TO_Q) or not (QUESTION_ATTENTION_BILINEAR)) 


   local x = nn.Identity()()
   local xemb
   assert(not(neatQA.DO_TRAINING_ON_ATTENTION_EMBEDDINGS))
   xemb = nn.BlockGradientLayer(params.batch_size, params.embeddings_dimensionality)(nn.LookupTableMaskZero(params.vocab_size,params.embeddings_dimensionality)(x))

   local y = nn.Identity()()
   local z = nn.Identity()()
   local u = nn.Identity()()

   local questionHistory = nn.Identity()()

   local position = nn.Identity()()

   local lastFixHistory = nn.Identity()()

------------
print("At 628: Warning: Doing ablation")
local z2 = nn.MulConstant(0.0)(nn.Linear(params.rnn_size,1)(z))
local u2 = nn.MulConstant(0.0)(nn.Linear(params.rnn_size,1)(u))

-----------
xembOrig = xemb



if false then
 assert(not(attention.ALLOW_HIDDEN_LAYERS))
 print("Using intermediate vector representation")
 embeddingIntermediateDimension = 50
 xemb = nn.ELU()(nn.Linear(100,embeddingIntermediateDimension)(xemb))
else
 print("Using direct input")
 xemb = xemb
 embeddingIntermediateDimension = 100
end

---------------------------------------
---------------------------------------



local dimensionOfConvolution

local number_of_inputs = nil
y0 = y

   questionTokensEmbeddings = y0
local inputReplicated

  inputReplicated = nn.ReplicateDynamic()({questionTokensEmbeddings,xembOrig})
  inputReplicated = nn.View(-1,params.embeddings_dimensionality)(inputReplicated)
  questionTokensEmbeddings = nn.View(-1,params.embeddings_dimensionality)(questionTokensEmbeddings)

QUESTION_ATTENTION_BILINEAR = false --true --false --true--true --false --true --false --true
print("Question Attention bilinear? ")
print(QUESTION_ATTENTION_BILINEAR)

   dimensionOfConvolution = 1
     questionTokensEmbeddings = nn.DotProduct()({(inputReplicated), (questionTokensEmbeddings)})
assert(dimensionOfConvolution == 1)
questionTokensEmbeddingsFlat = nn.View(params.batch_size,-1,dimensionOfConvolution)(questionTokensEmbeddings) -- batch * length * conv
   questionEmbeddingMax = nn.Max(2,3)(questionTokensEmbeddingsFlat)
   questionEmbeddingMax = nn.View(params.batch_size,dimensionOfConvolution)(questionEmbeddingMax) -- batch * conv
fromInput = nn.JoinTable(1,1)({xemb,(z2),(u2)})




DOING_VARIANT = 59
--DOING_SIGMOID_ON_GATE = true
print("VARIANT NUMBER "..DOING_VARIANT)
if DOING_VARIANT == 54 or DOING_VARIANT == 58 then --54, 58
  assert(false)
  print("DOING 54 58 at 10521")
  fromInput = nn.Linear(30,1)(nn.Dropout(0.1)(nn.ReLU()(nn.Linear(102,30)(fromInput)))) -- attention score from the input
  gateFromInput = (nn.Linear(30,1)(nn.Dropout(0.1)(nn.ReLU()(nn.Linear(embeddingIntermediateDimension,30)(xemb))))) -- gating score from the input
 if DOING_VARIANT == 54 then -- 54
   gateFromInput = nn.Sigmoid()(gateFromInput)
 elseif DOING_VARIANT == 58 then
   print("No sigmoid for gate.")
 else
   assert(false)
 end
else --55, 56, 57, 59
  assert(DOING_VARIANT == 55 or DOING_VARIANT == 59 or DOING_VARIANT == 56)
  print("DOING 55 56 57 at 10927")
  fromInput = attention.scalarCoefficient(embeddingIntermediateDimension+2,fromInput,"fromInput") -- attention score from the input
  gateFromInput = attention.scalarCoefficient(embeddingIntermediateDimension,xemb,"gateFromInput") -- gating score from the input
 if DOING_VARIANT == 55 or DOING_VARIANT == 56 then -- 55,56,57
   gateFromInput = nn.Sigmoid()(gateFromInput)
 else
  assert(DOING_VARIANT == 59)
   print("No sigmoid for gate.")
 end
end

if neatQA.STORE_Q_ATTENTION then
   questionEmbeddingMax = nn.StoreInContainerLayer("attRelativeToQ", true)(questionEmbeddingMax)
   fromInput = nn.StoreInContainerLayer("fromInput", true)(fromInput)
   gateFromInput = nn.StoreInContainerLayer("gateFromInput", true)(gateFromInput)
end

if false then
  print("Using non-binary original Q similarity, centered at 0.7")
  dotproductAffine = nn.AddConstant(-0.6599)(questionEmbeddingMax)
elseif false then
  print("Using non-binary original Q similarity, not centered")
  dotproductAffine = questionEmbeddingMax
elseif false then
  print("Taking binary variable, not centered")
  dotproductAffine = nn.AddConstant(0.0)(nn.Threshold(0.99,0.0)(questionEmbeddingMax))
else
  print("Taking binary variable, centered")
  dotproductAffine = nn.AddConstant(-0.116)(nn.Threshold(0.99,0.0)(questionEmbeddingMax))
--  dotproductAffine = nn.Threshold(0.99,0.0)(questionEmbeddingMax)
end
if neatQA.STORE_Q_ATTENTION then
   dotproductAffine = nn.StoreInContainerLayer("dotproductAffine", true)(dotproductAffine)
end


-- from Q
gatedDotProduct = nn.CMulTable()({gateFromInput, dotproductAffine})




assert(dimensionOfConvolution == 1)
print(dimensionOfConvolution)



---------------------------------------
---------------------------------------




local qSimForHistory
if false and (DOING_VARIANT == 56 or DOING_VARIANT == 59) then
  assert(false)
  print("Using raw similarity for history")
  qSimForHistory = nn.AddConstant(-0.6599)(questionEmbeddingMax)
else
  print("Using binary similarity for history")
  qSimForHistory = dotproductAffine
end

if false then
  assert(false)
  print("Fixed decay; dependent on source word")
  -- weight for the Quest value for future
  questHistOutGate = nn.Linear(embeddingIntermediateDimension,1)(xemb)
  questHistOutGate = nn.StoreInContainerLayer("questHistoryOutGate", true)(questHistOutGate)
  gatedForHistory = nn.CMulTable()({questHistOutGate, qSimForHistory})
  gatedForHistory = nn.StoreInContainerLayer("gatedFromHistory", true)(gatedForHistory)


  -- decay factor
  questHistFutureGate = nn.Sigmoid()(nn.Add(-1,true)(nn.MulConstant(0.0)(qSimForHistory)))  --     nn.Linear(100,1)(xemb)) -- how much should come from this word
  questHistFutureGate = nn.StoreInContainerLayer("questHistoryFutureGate", true)(questHistFutureGate)
  gatedQuestForFuture = nn.CAddTable()({nn.CMulTable()({questHistFutureGate, gatedForHistory}), nn.CMulTable()({nn.SAdd(-1,true)(questHistFutureGate), questionHistory})})
  gatedQuestForFuture = nn.StoreInContainerLayer("gatedQuestForFuture", true)(gatedQuestForFuture)

  ------------------
  -- no gate is applied to what has come from the previous words
  gatedFromHistory = questionHistory
elseif true then
  print("Fixed decay; dependent on target word")

  -- TODO do not add contribution for this word when skipping

  -- decay factor
  questHistFutureGate = nn.Sigmoid()(nn.Add(-1,true)(nn.MulConstant(0.0)(qSimForHistory)))  --     nn.Linear(100,1)(xemb)) -- how much should come from this word
  questHistFutureGate = nn.StoreInContainerLayer("questHistoryFutureGate", true)(questHistFutureGate)


--  gatedQuestForFuture = nn.CAddTable()({nn.CMulTable()({questHistFutureGate, qSimForHistory}), nn.CMulTable()({nn.SAdd(-1,true)(questHistFutureGate), questionHistory})})
--  gatedQuestForFuture = nn.StoreInContainerLayer("gatedQuestForFuture", true)(gatedQuestForFuture)

  -- weight for the Quest value for future
  questHistOutGate = nn.Linear(embeddingIntermediateDimension,1)(xemb)
  questHistOutGate = nn.StoreInContainerLayer("questHistoryOutGate", true)(questHistOutGate)

  -- used for this decision
  gatedFromHistory = nn.CMulTable()({questHistOutGate, questionHistory})
  gatedFromHistory = nn.StoreInContainerLayer("gatedFromHistory", true)(gatedFromHistory)
else
  assert(false)
  print("Varied decay and dependent on target word")
  questHistFutureGate = nn.Sigmoid()(nn.Linear(embeddingIntermediateDimension,1)(xemb)) -- how much should come from this word
  questHistFutureGate = nn.StoreInContainerLayer("questHistoryFutureGate", true)(questHistFutureGate)
  gatedQuestForFuture = nn.CAddTable()({nn.CMulTable()({questHistFutureGate, qSimForHistory}), nn.CMulTable()({nn.SAdd(-1,true)(questHistFutureGate), questionHistory})})
  gatedQuestForFuture = nn.StoreInContainerLayer("gatedQuestForFuture", true)(gatedQuestForFuture)

  questHistOutGate = nn.Linear(embeddingIntermediateDimension,1)(xemb)
  if DOING_VARIANT == 55 then
    print("Sigmoid for questHistOutGate")
    questHistOutGate = nn.Sigmoid()(questHistOutGate)
  end
  questHistOutGate = nn.StoreInContainerLayer("questHistoryOutGate", true)(questHistOutGate)
  gatedFromHistory = nn.CMulTable()({questHistOutGate, questionHistory})
  gatedFromHistory = nn.StoreInContainerLayer("gatedFromHistory", true)(gatedFromHistory)
end



---------------------------------------
---------------------------------------



positionGate = nn.Linear(embeddingIntermediateDimension,1)(xemb)
positionGate = nn.StoreInContainerLayer("positionGate", true)(positionGate)
local centeredPosition = nn.AddConstant(-0.45)(position)
positionGated = nn.CMulTable()({positionGate, centeredPosition})
positionGated = nn.StoreInContainerLayer("positionGated", true)(positionGated)




---------------------------------------
---------------------------------------

  -- output gate
if false then
  local gateLastFixHistory = nn.Linear(embeddingIntermediateDimension,1)(xemb)
  local lastFixHistoryInContainer = nn.StoreInContainerLayer("lastFixHistory", true)(lastFixHistory)
  gateLastFixHistory = nn.StoreInContainerLayer("gateLastFixHistory", true)(gateLastFixHistory)
  local decayLastFix = nn.Sigmoid()(nn.Add(-1,true)(nn.MulConstant(0.0)(qSimForHistory)))
  local gatedLastFix = nn.CMulTable()({gateLastFixHistory, lastFixHistoryInContainer})
end





local gatedCondition
local gatedConditionTimesPosition

--
assert(DOING_EVALUATION_OUTPUT or neatQA.CONDITION == "mixed" or neatQA.CONDITION == "fullpreview" or neatQA.CONDITION == "fullnopreview")

  local centeredCondition = nn.AddConstant(-0.5)(condition_mask)
  local conditionGate = nn.Linear(embeddingIntermediateDimension,1)(xemb)

  conditionGate = nn.StoreInContainerLayer("conditionGate", true)(conditionGate)

  gatedCondition = nn.CMulTable()({centeredCondition, conditionGate})

  gatedDotProduct = nn.CMulTable()({condition_mask,gatedDotProduct})

  -- this is used in the decision
  gatedFromHistory = nn.CMulTable()({condition_mask,gatedFromHistory})

  local conditionTimesPosition = nn.CMulTable()({centeredCondition, centeredPosition})
  
  local conditionTimesPositionGate = nn.Linear(embeddingIntermediateDimension,1)(xemb)
  conditionTimesPositionGate = nn.StoreInContainerLayer("conditionTimesPositionGate",true)(conditionTimesPositionGate)
  gatedConditionTimesPosition = nn.CMulTable()({conditionTimesPositionGate, conditionTimesPosition})





---------------------------------------
---------------------------------------


if neatQA.CONDITION == "fullpreview" or neatQA.CONDITION == "fullnopreview" then
  print("Warning: neutralize interactions with condition! at 350.63")
  gatedConditionTimesPosition = nn.MulConstant(0.0)(gatedConditionTimesPosition)
  gatedCondition = nn.MulConstant(0.0)(gatedCondition)
end



local linearTerms
linearTerms = {gatedDotProduct, fromInput, gatedFromHistory, positionGated}
if true then
  print("Warning: ignoring LastFix")
   gatedLastFix = nn.MulConstant(0.0)(lastFixHistory)
if false then
  gatedLastFix = nn.MulConstant(0.0)(gatedLastFix)
end
end

if true then
  table.insert(linearTerms, gatedLastFix)

end
if true or neatQA.CONDITION == "mixed" then
  table.insert(linearTerms, gatedCondition)
  table.insert(linearTerms, gatedConditionTimesPosition)
end
if false then
 table.insert(linearTerms, nn.CMulTable()({questionHistoryFromSource, condition_mask})  )
end
local bilinear = nn.CAddTable()(linearTerms)



---------------------------------------
---------------------------------------


local attentionBias = 0.0
print("attention bias"..attentionBias)
bilinear = nn.AddConstant(attentionBias)(bilinear)



---------------------------------------
---------------------------------------


local decisionsAndProbs
local attention
if neatQA.better_logsigmoid_gradients then
   decisionsAndProbs = nn.BernoulliLayerSigmoidWithLogProb(params.batch_size,1)(bilinear)
   attention = nn.SelectTable(3)(decisionsAndProbs)
else
   attention = nn.Sigmoid()(bilinear)
   decisionsAndProbs = nn.BernoulliLayer(params.batch_size,1)(attention)
end

   local decisions = nn.SelectTable(1)(decisionsAndProbs)
   local probs = nn.SelectTable(2)(decisionsAndProbs)
--probs = nn.PrintLayer("probs",1.0,true)(probs)
   local attendedInputTensors = nn.CMulTable()({x,decisions})

local blockedAttention
if neatQA.ANALYTICAL_MINIMIZATION_OF_FIXATIONS then
  blockedAttention = attention
else
  blockedAttention = nn.BlockGradientLayer(params.batch_size,1)(attention)
end
local blockedDecisions = nn.BlockGradientLayer(params.batch_size,1)(nn.View(params.batch_size,1)(decisions))
local blockedInput = nn.BlockGradientLayer(params.batch_size,1)(attendedInputTensors)


  print("Ignoring qSimForHistory if the word is skipped. This is innovation for 109 (April 24, 2017)")
  -- ignore qSimForHistory if the word is skipped
  questHistFutureGate = nn.CMulTable()({decisions,questHistFutureGate})
  gatedQuestForFuture = nn.CAddTable()({nn.CMulTable()({questHistFutureGate, qSimForHistory}), nn.CMulTable()({nn.SAdd(-1,true)(questHistFutureGate), questionHistory})})
  gatedQuestForFuture = nn.StoreInContainerLayer("gatedQuestForFuture", true)(gatedQuestForFuture)

  local lastFixForFuture  = lastFixHistory
if false then
   lastFixForFuture = nn.CAddTable()({nn.CMulTable()({decayLastFix, nn.AddConstant(-0.45)(decisions)}), nn.CMulTable()({nn.SAdd(-1,true)(decayLastFix), lastFixHistoryInContainer})})
end



   local inputNodes = {x}
     table.insert(inputNodes, y)
   table.insert(inputNodes,z)
   table.insert(inputNodes, u)
   table.insert(inputNodes, questionHistory)
   table.insert(inputNodes, position)
   table.insert(inputNodes, lastFixHistory)
   if true or  neatQA.CONDITION == "mixed" then
     table.insert(inputNodes,condition_mask)
   end
   local module = nn.gModule(inputNodes,{blockedAttention,blockedDecisions,probs,blockedInput, gatedQuestForFuture,lastFixForFuture })


  parameters, gradparameters = module:parameters()
  for i=1, #parameters do
    local epsilon = math.sqrt(6.0/torch.LongTensor(parameters[i]:size()):sum())
    parameters[i]:uniform(-epsilon, epsilon)
    gradparameters[i]:zero()
    if parameters[i]:storage():size() == 1 then
      print("Note: singleton parameters initialized to zero")
      parameters[i]:zero()
    end
    print(i.."  "..parameters[i]:norm().." at 16743 ")
    if parameters[i]:dim() > 1 then
        print("   "..parameters[i][1]:norm())
    end
    if false and (i==4) then
      assert(DOING_DEBUGGING)
      parameters[i]:zero()
    end

  end
--  if false and #param == 1 then
--    param = {"PLACEHOLDER","PLACEHOLDER",param[1]}
--    gradparam = {"PLACEHOLDER","PLACEHOLDER",gradparam[1]}
--  end

if (not IS_CONTINUING_ATTENTION) then
  assert(#param == 1)
end


  if #param ~= #parameters then
    print("WARNING: attention parameters have different numbers")
    print("Loaded:")
    print(param)
    print("Own:")
    print(parameters)
    print("===")
    if (#param == 10) then
      for j=4,9 do
         param[j] = param[j+1]
         gradparam[j] = gradparam[j+1]
      end
      param[10] = nil
      gradparam[10] = nil
      print("Have removed constant offset by removing element 4")
      print("Loaded:")
      print(param)
    end
  end
  if DOING_EVALUATION_OUTPUT then
     print(param)
     print(parameters)
     assert(#param == #parameters)
  end
  for i=1,#param do
    if param[i] ~= "PLACEHOLDER" then
      if param[i]:nElement() == parameters[i]:nElement() then
        parameters[i]:copy(param[i])
        if false then
          print("Not copying gradparam at 17342")
          gradparameters[i]:copy(gradparam[i]):zero()
        end
      else
        print("Not copying parameter "..i.." because of incompatible dimensions.")
        print(param[i]:nElement())
        print(parameters[i]:nElement())
        assert(not(DOING_EVALUATION_OUTPUT))
      end
--if false then
--   print("WARNING: Setting singleton parameters of loaded model to zero!")
--    if parameters[i]:storage():size() == 1 then
--        parameters[i]:zero()
--    end
--    print(parameters[i]:storage():size().."  "..parameters[i]:norm())
--end
    end
  end


if true then
  params.CONDITION_BIAS = nil
  print("WARNING: BLOCKING CONDITION BIAS")
end
if DOING_EVALUATION_OUTPUT and (params.CONDITION_BIAS ~= nil) then
  print("APPLYING CONDITION BIAS "..params.CONDITION_BIAS)
  parameters[17][1] = parameters[17][1] + params.CONDITION_BIAS
end


  for i=1, #parameters do
    print("# "..i..parameters[i]:norm().." at 16743 ")
    if parameters[i]:dim() > 1 then
        print("#   "..parameters[i][1]:norm())
    end
  end


  return transfer_data(module)
end

































































































































-----------------------------------------------
-----------------------------------------------
-----------------------------------------------







-----------------------------------------------
-----------------------------------------------
--------------------------------------------------

--------------------------------------------------

--------------------------------------------------

--------------------------------------------------

--------------------------------------------------

function attention.createAttentionNewCombinationReduceCollinearityNotCleanedUp(param,gradparam)

 assert(false) -- July 30, 2017

 assert(neatQA.CONDITION ~= "nopreview")

 local condition_mask
 if true or neatQA.CONDITION == "mixed" then
   condition_mask = nn.Identity()()
   --current_condition = ["preview", "nopreview"][1+torch.bernoulli(0.5)]
 else
   --current_condition = neatQA.CONDITION
 end
 --assert(current_condition == "preview" or current_condition == "nopreview")



assert(neatQA.CENTER_PREDICTORS)
 print("Doing biaffine with reduced collinearity at 3754")
assert(neatQA.NEW_COMBINATION)
   assert(neatQA.ATTENTION_DOES_Q_ATTENTION)

   assert(params.TASK == 'neat-qa')


   print("Doing training on attention embeddings?")
   print(neatQA.DO_TRAINING_ON_ATTENTION_EMBEDDINGS)

APPLY_TRAFO_ALSO_TO_Q = false
print("APPLY_TRAFO_ALSO_TO_Q   "..tostring(APPLY_TRAFO_ALSO_TO_Q))
assert(not(neatQA.STRETCH_Q_EMBEDDINGS) or not(APPLY_TRAFO_ALSO_TO_Q))
assert(not(APPLY_TRAFO_ALSO_TO_Q) or not (QUESTION_ATTENTION_BILINEAR)) 


   local x = nn.Identity()()
   local xemb
   assert(not(neatQA.DO_TRAINING_ON_ATTENTION_EMBEDDINGS))
   xemb = nn.BlockGradientLayer(params.batch_size, params.embeddings_dimensionality)(nn.LookupTableMaskZero(params.vocab_size,params.embeddings_dimensionality)(x))

   local y = nn.Identity()()
   local z = nn.Identity()()
   local u = nn.Identity()()

   local questionHistory = nn.Identity()()

   local position = nn.Identity()()
------------
print("At 628: Warning: Doing ablation")
local z2 = nn.MulConstant(0.0)(nn.Linear(params.rnn_size,1)(z))
local u2 = nn.MulConstant(0.0)(nn.Linear(params.rnn_size,1)(u))

-----------
xembOrig = xemb



local dimensionOfConvolution

local number_of_inputs = nil
y0 = y

   questionTokensEmbeddings = y0
local inputReplicated

  inputReplicated = nn.ReplicateDynamic()({questionTokensEmbeddings,xembOrig})
  inputReplicated = nn.View(-1,params.embeddings_dimensionality)(inputReplicated)
  questionTokensEmbeddings = nn.View(-1,params.embeddings_dimensionality)(questionTokensEmbeddings)

QUESTION_ATTENTION_BILINEAR = false --true --false --true--true --false --true --false --true
print("Question Attention bilinear? ")
print(QUESTION_ATTENTION_BILINEAR)

   dimensionOfConvolution = 1
     questionTokensEmbeddings = nn.DotProduct()({(inputReplicated), (questionTokensEmbeddings)})
assert(dimensionOfConvolution == 1)
questionTokensEmbeddingsFlat = nn.View(params.batch_size,-1,dimensionOfConvolution)(questionTokensEmbeddings) -- batch * length * conv
   questionEmbeddingMax = nn.Max(2,3)(questionTokensEmbeddingsFlat)
   questionEmbeddingMax = nn.View(params.batch_size,dimensionOfConvolution)(questionEmbeddingMax) -- batch * conv
fromInput = nn.JoinTable(1,1)({xemb,(z2),(u2)})




DOING_VARIANT = 59
--DOING_SIGMOID_ON_GATE = true
print("VARIANT NUMBER "..DOING_VARIANT)
if DOING_VARIANT == 54 or DOING_VARIANT == 58 then --54, 58
  assert(false)
  print("DOING 54 58 at 10521")
  fromInput = nn.Linear(30,1)(nn.Dropout(0.1)(nn.ReLU()(nn.Linear(102,30)(fromInput)))) -- attention score from the input
  gateFromInput = (nn.Linear(30,1)(nn.Dropout(0.1)(nn.ReLU()(nn.Linear(100,30)(xemb))))) -- gating score from the input
 if DOING_VARIANT == 54 then -- 54
   gateFromInput = nn.Sigmoid()(gateFromInput)
 elseif DOING_VARIANT == 58 then
   print("No sigmoid for gate.")
 else
   assert(false)
 end
else --55, 56, 57, 59
  assert(DOING_VARIANT == 55 or DOING_VARIANT == 59 or DOING_VARIANT == 56)
  print("DOING 55 56 57 at 10927")
  fromInput = nn.Linear(102,1)(fromInput) -- attention score from the input
  gateFromInput = (nn.Linear(100,1)(xemb)) -- gating score from the input
 if DOING_VARIANT == 55 or DOING_VARIANT == 56 then -- 55,56,57
   gateFromInput = nn.Sigmoid()(gateFromInput)
 else
  assert(DOING_VARIANT == 59)
   print("No sigmoid for gate.")
 end
end

if neatQA.STORE_Q_ATTENTION then
   questionEmbeddingMax = nn.StoreInContainerLayer("attRelativeToQ", true)(questionEmbeddingMax)
   fromInput = nn.StoreInContainerLayer("fromInput", true)(fromInput)
   gateFromInput = nn.StoreInContainerLayer("gateFromInput", true)(gateFromInput)
end

if false then
  print("Using non-binary original Q similarity, centered at 0.7")
  dotproductAffine = nn.AddConstant(-0.6599)(questionEmbeddingMax)
elseif false then
  print("Using non-binary original Q similarity, not centered")
  dotproductAffine = questionEmbeddingMax
elseif false then
  print("Taking binary variable, not centered")
  dotproductAffine = nn.AddConstant(0.0)(nn.Threshold(0.99,0.0)(questionEmbeddingMax))
else
  print("Taking binary variable, centered")
  dotproductAffine = nn.AddConstant(-0.116)(nn.Threshold(0.99,0.0)(questionEmbeddingMax))
--  dotproductAffine = nn.Threshold(0.99,0.0)(questionEmbeddingMax)
end
if neatQA.STORE_Q_ATTENTION then
   dotproductAffine = nn.StoreInContainerLayer("dotproductAffine", true)(dotproductAffine)
end


-- from Q
gatedDotProduct = nn.CMulTable()({gateFromInput, dotproductAffine})




assert(dimensionOfConvolution == 1)
print(dimensionOfConvolution)

local qSimForHistory
if false and (DOING_VARIANT == 56 or DOING_VARIANT == 59) then
  print("Using raw similarity for history")
  qSimForHistory = nn.AddConstant(-0.6599)(questionEmbeddingMax)
else
  print("Using binary similarity for history")
  qSimForHistory = dotproductAffine
end

if false then
  print("Fixed decay; dependent on source word")
  -- weight for the Quest value for future
  questHistOutGate = nn.Linear(100,1)(xemb)
  questHistOutGate = nn.StoreInContainerLayer("questHistoryOutGate", true)(questHistOutGate)
  gatedForHistory = nn.CMulTable()({questHistOutGate, qSimForHistory})
  gatedForHistory = nn.StoreInContainerLayer("gatedFromHistory", true)(gatedForHistory)


  -- decay factor
  questHistFutureGate = nn.Sigmoid()(nn.Add(-1,true)(nn.MulConstant(0.0)(qSimForHistory)))  --     nn.Linear(100,1)(xemb)) -- how much should come from this word
  questHistFutureGate = nn.StoreInContainerLayer("questHistoryFutureGate", true)(questHistFutureGate)
  gatedQuestForFuture = nn.CAddTable()({nn.CMulTable()({questHistFutureGate, gatedForHistory}), nn.CMulTable()({nn.SAdd(-1,true)(questHistFutureGate), questionHistory})})
  gatedQuestForFuture = nn.StoreInContainerLayer("gatedQuestForFuture", true)(gatedQuestForFuture)

  ------------------
  -- no gate is applied to what has come from the previous words
  gatedFromHistory = questionHistory
elseif true then
  print("Fixed decay; dependent on target word")

  -- TODO do not add contribution for this word when skipping

  -- decay factor
  questHistFutureGate = nn.Sigmoid()(nn.Add(-1,true)(nn.MulConstant(0.0)(qSimForHistory)))  --     nn.Linear(100,1)(xemb)) -- how much should come from this word
  questHistFutureGate = nn.StoreInContainerLayer("questHistoryFutureGate", true)(questHistFutureGate)
  gatedQuestForFuture = nn.CAddTable()({nn.CMulTable()({questHistFutureGate, qSimForHistory}), nn.CMulTable()({nn.SAdd(-1,true)(questHistFutureGate), questionHistory})})
  gatedQuestForFuture = nn.StoreInContainerLayer("gatedQuestForFuture", true)(gatedQuestForFuture)


  -- weight for the Quest value for future
  questHistOutGate = nn.Linear(100,1)(xemb)
  questHistOutGate = nn.StoreInContainerLayer("questHistoryOutGate", true)(questHistOutGate)

  -- used for this decision
  gatedFromHistory = nn.CMulTable()({questHistOutGate, questionHistory})
  gatedFromHistory = nn.StoreInContainerLayer("gatedFromHistory", true)(gatedFromHistory)





else
  print("Varied decay and dependent on target word")
  questHistFutureGate = nn.Sigmoid()(nn.Linear(100,1)(xemb)) -- how much should come from this word
  questHistFutureGate = nn.StoreInContainerLayer("questHistoryFutureGate", true)(questHistFutureGate)
  gatedQuestForFuture = nn.CAddTable()({nn.CMulTable()({questHistFutureGate, qSimForHistory}), nn.CMulTable()({nn.SAdd(-1,true)(questHistFutureGate), questionHistory})})
  gatedQuestForFuture = nn.StoreInContainerLayer("gatedQuestForFuture", true)(gatedQuestForFuture)

  questHistOutGate = nn.Linear(100,1)(xemb)
  if DOING_VARIANT == 55 then
    print("Sigmoid for questHistOutGate")
    questHistOutGate = nn.Sigmoid()(questHistOutGate)
  end
  questHistOutGate = nn.StoreInContainerLayer("questHistoryOutGate", true)(questHistOutGate)
  gatedFromHistory = nn.CMulTable()({questHistOutGate, questionHistory})
  gatedFromHistory = nn.StoreInContainerLayer("gatedFromHistory", true)(gatedFromHistory)
end

positionGate = nn.Linear(100,1)(xemb)
positionGate = nn.StoreInContainerLayer("positionGate", true)(positionGate)
local centeredPosition = nn.AddConstant(-0.45)(position)
positionGated = nn.CMulTable()({positionGate, centeredPosition})
positionGated = nn.StoreInContainerLayer("positionGated", true)(positionGated)


--questionHistory
local gatedCondition
local gatedConditionTimesPosition

if neatQA.CONDITION == "nopreview" then
  gatedDotProduct = nn.MulConstant(0.0)(gatedDotProduct)
  gatedFromHistory = nn.MulConstant(0.0)(gatedFromHistory)
elseif true or neatQA.CONDITION == "mixed" then
  local centeredCondition = nn.AddConstant(-0.5)(condition_mask)
  local conditionGate = nn.Linear(100,1)(xemb)

  conditionGate = nn.StoreInContainerLayer("conditionGate", true)(conditionGate)

  gatedCondition = nn.CMulTable()({centeredCondition, conditionGate})

  gatedDotProduct = nn.CMulTable()({condition_mask,gatedDotProduct})

  -- this is used in the decision
  gatedFromHistory = nn.CMulTable()({condition_mask,gatedFromHistory})

  local conditionTimesPosition = nn.CMulTable()({centeredCondition, centeredPosition})
  
  local conditionTimesPositionGate = nn.Linear(100,1)(xemb)
  conditionTimesPositionGate = nn.StoreInContainerLayer("conditionTimesPositionGate",true)(conditionTimesPositionGate)
  gatedConditionTimesPosition = nn.CMulTable()({conditionTimesPositionGate, conditionTimesPosition})

else
 assert(neatQA.CONDITION == "preview")
end

local linearTerms
linearTerms = {gatedDotProduct, fromInput, gatedFromHistory, positionGated}
if true or neatQA.CONDITION == "mixed" then
  table.insert(linearTerms, gatedCondition)
  table.insert(linearTerms, gatedConditionTimesPosition)
end

local bilinear = nn.CAddTable()(linearTerms)






local attentionBias = 0.0
print("attention bias"..attentionBias)
bilinear = nn.AddConstant(attentionBias)(bilinear)


local decisionsAndProbs
local attention
if neatQA.better_logsigmoid_gradients then
   decisionsAndProbs = nn.BernoulliLayerSigmoidWithLogProb(params.batch_size,1)(bilinear)
   attention = nn.SelectTable(3)(decisionsAndProbs)
else
   attention = nn.Sigmoid()(bilinear)
   decisionsAndProbs = nn.BernoulliLayer(params.batch_size,1)(attention)
end

   local decisions = nn.SelectTable(1)(decisionsAndProbs)
   local probs = nn.SelectTable(2)(decisionsAndProbs)
--probs = nn.PrintLayer("probs",1.0,true)(probs)
   local attendedInputTensors = nn.CMulTable()({x,decisions})

local blockedAttention
if neatQA.ANALYTICAL_MINIMIZATION_OF_FIXATIONS then
  blockedAttention = attention
else
  blockedAttention = nn.BlockGradientLayer(params.batch_size,1)(attention)
end
local blockedDecisions = nn.BlockGradientLayer(params.batch_size,1)(nn.View(params.batch_size,1)(decisions))
local blockedInput = nn.BlockGradientLayer(params.batch_size,1)(attendedInputTensors)



   local inputNodes = {x}
     table.insert(inputNodes, y)
   table.insert(inputNodes,z)
   table.insert(inputNodes, u)
   table.insert(inputNodes, questionHistory)
   table.insert(inputNodes, position)
   if true or neatQA.CONDITION == "mixed" then
     table.insert(inputNodes,condition_mask)
   end
   local module = nn.gModule(inputNodes,{blockedAttention,blockedDecisions,probs,blockedInput, gatedQuestForFuture})


  parameters, gradparameters = module:parameters()
  for i=1, #parameters do
    local epsilon = math.sqrt(6.0/torch.LongTensor(parameters[i]:size()):sum())
    parameters[i]:uniform(-epsilon, epsilon)
    gradparameters[i]:zero()
    if parameters[i]:storage():size() == 1 then
      print("Note: singleton parameters initialized to zero")
      parameters[i]:zero()
    end
    print(i.."  "..parameters[i]:norm().." at 16743 ")
    if parameters[i]:dim() > 1 then
        print("   "..parameters[i][1]:norm())
    end
    if false and (i==4) then
      assert(DOING_DEBUGGING)
      parameters[i]:zero()
    end

  end
--  if false and #param == 1 then
--    param = {"PLACEHOLDER","PLACEHOLDER",param[1]}
--    gradparam = {"PLACEHOLDER","PLACEHOLDER",gradparam[1]}
--  end


  if #param ~= #parameters then
    print("WARNING: attention parameters have different numbers")
    print("Loaded:")
    print(param)
    print("Own:")
    print(parameters)
    print("===")
    if (#param == 10) then
      for j=4,9 do
         param[j] = param[j+1]
         gradparam[j] = gradparam[j+1]
      end
      param[10] = nil
      gradparam[10] = nil
      print("Have removed constant offset by removing element 4")
      print("Loaded:")
      print(param)
    end
  end
  for i=1,#param do
    if param[i] ~= "PLACEHOLDER" then
      if param[i]:nElement() == parameters[i]:nElement() then
        parameters[i]:copy(param[i])
        if false then
          print("Not copying gradparam at 17342")
          gradparameters[i]:copy(gradparam[i]):zero()
        end
      else
        print("Not copying parameter "..i.." because of incompatible dimensions.")
        print(param[i]:nElement())
        print(parameters[i]:nElement())
      end
--if false then
--   print("WARNING: Setting singleton parameters of loaded model to zero!")
--    if parameters[i]:storage():size() == 1 then
--        parameters[i]:zero()
--    end
--    print(parameters[i]:storage():size().."  "..parameters[i]:norm())
--end
    end
  end

  for i=1, #parameters do
    print("# "..i..parameters[i]:norm().." at 16743 ")
    if parameters[i]:dim() > 1 then
        print("#   "..parameters[i][1]:norm())
    end
  end


  return transfer_data(module)
end






--------------------------------------------------

--------------------------------------------------















---------------------------

---------------------------

---------------------------

---------------------------






function attention.createAttentionNewCombination(param,gradparam)
if neatQA.CENTER_PREDICTORS then
  return attention.createAttentionNewCombinationReduceCollinearity(param,gradparam)
end
end






function attention.createAttentionBilinearSimple(param,gradparam)
 print("23911 simple bilinear (55)")
 print("Doing simple version only depending on the word")
 if neatQA.USE_INNOVATIVE_ATTENTION then
   print("Usng innovative attention!")
   return attention.createAttentionBilinearEXPERIMENTINGDirect(param,gradparam)
 end
end





function attention.createAttentionBilinear(param,gradparam)
if neatQA.NEW_COMBINATION then
  return attention.createAttentionNewCombination(param,gradparam)
end
end
























