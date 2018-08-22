autoencoding = {}
autoencoding.__name = "autoencoding"

autoencoding.USE_PRETRAINED_EMBEDDINGS = true

print(autoencoding)

function autoencoding.create_multilayer_network(withOutput, doZeroMaskingOnLookupTable, inputDropout, numberOfLayers)
assert(false)
  local x                = nn.Identity()()
  local i

  -- somehow seems to be harmful, so don't use!
  if doZeroMaskingOnLookupTable then
    i = nn.LookupTableMaskZero(params.vocab_size,params.embeddings_dimensionality)(x)
  else
    i = nn.LookupTable(params.vocab_size,params.embeddings_dimensionality)(x)
  end

  local next_s           = {}

  local inputToNextLayer = i

  local next_c = {}
  local next_h = {}
 
  local inputNodes = {x}
  local outputNodes = {}

  local next_c_l
  local next_h_l

  for layer = 1,numberOfLayers do

    local prev_c_l = nn.Identity()()
    local prev_h_l = nn.Identity()()

    table.insert(inputNodes, prev_c_l)
    table.insert(inputNodes, prev_h_l)


    local dimensionalityOfInput
    if (layer == 1) then
        dimensionalityOfInput = params.embeddings_dimensionality
    else
        inputToNextLayer = next_c_l
--        inputToNextLayer = nn.JoinTable(1,2)({i,next_c_l})
        dimensionalityOfInput = params.rnn_size
    end


    if inputDropout == true then
      inputToNextLayer = nn.Dropout(0.2)(inputToNextLayer)
    end

    next_c_l, next_h_l = lstm.lstm(inputToNextLayer, prev_c_l, prev_h_l, dimensionalityOfInput)
    table.insert(outputNodes, next_c_l)
    table.insert(outputNodes, next_h_l)
  end

  if withOutput  then
        local h2y              = nn.Linear(params.rnn_size, params.vocab_size)(tableOfOutputNodes_C[numberOfLayers])
        local output = nn.MulConstant(-1)(nn.LogSoftMax()(h2y))
        table.insert(outputNodes,output)
  end
  local module = nn.gModule(inputNodes,outputNodes)

  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end







function autoencoding.create_network(withOutput, doZeroMaskingOnLookupTable, inputDropout)
  local x                = nn.Identity()()
  local prev_c           = nn.Identity()()
  local prev_h           = nn.Identity()()
  local i

  if doZeroMaskingOnLookupTable then
    i = nn.LookupTableMaskZero(params.vocab_size,params.embeddings_dimensionality)(x)
  else
    i = nn.LookupTable(params.vocab_size,params.embeddings_dimensionality)(x)
  end

  if inputDropout == true then
    i = nn.Dropout(0.2)(i)
  end

  local next_s           = {}
  local next_c, next_h = lstm.lstm(i, prev_c, prev_h, params.embeddings_dimensionality)
  local module
  if withOutput  then
        local h2y              = nn.Linear(params.rnn_size, params.vocab_size)(next_c)
        local output = nn.MulConstant(-1)(nn.LogSoftMax()(h2y))
      module = nn.gModule({x, prev_c, prev_h},
                                      {next_c, next_h, output})
  else
      module = nn.gModule({x, prev_c, prev_h},
                                      {next_c, next_h})
  end

  parameters = module:parameters()
  for i=1, #parameters do
    local epsilon = math.sqrt(6.0/torch.LongTensor(parameters[i]:size()):sum())
    parameters[i]:uniform(-epsilon, epsilon)
  end

--  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function autoencoding.setupAutoencoding()
  print("Creating a RNN LSTM network.")

assert(false)
  -- initialize data structures
  -- TODO why is there no declaration for model.sA, model.sRA here?
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

  actor_c ={[0] = torch.CudaTensor(params.rnn_size)}
  actor_h = {[0] = torch.CudaTensor(params.rnn_size)}

  reader_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero() --TODO they have to be intiialized decently
  reader_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()

  if params.TASK == 'combined' then
     reader_output = {}
     surprisal_values = {[1] = transfer_data(torch.zeros(params.batch_size,1))}
  end


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

   if params.TASK == 'combined' then
      nll_reader = torch.FloatTensor(params.batch_size)
   end

  attention_inputTensors = {}
  if USE_PREDICTION_FOR_ATTENTION then
     for i=1, params.seq_length do
        attention_inputTensors[i] = torch.CudaTensor(params.batch_size)
     end
  end




  ones = transfer_data(torch.ones(params.batch_size))
  rewardBaseline = 0

   variance_average = 100
   recurrent_variance_average = 100
   if not LOAD then
     -- READER
     local reader_core_network
     if params.TASK == 'autoencoding' then
        reader_core_network = autoencoding.create_network(false)
     elseif params.TASK == 'combined' then
        reader_core_network = autoencoding.create_network(true,false,true)
     else
        crash()
     end



     if autoencoding.USE_PRETRAINED_EMBEDDINGS then
        local parameters, _ = reader_core_network:parameters()
        readDict.setToPretrainedEmbeddings(parameters[1])
     end


     paramxR, paramdxR = reader_core_network:getParameters()

     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network)
     end





     -- ACTOR
     local actor_core_network = autoencoding.create_network(true)

     if autoencoding.USE_PRETRAINED_EMBEDDINGS then
        local parameters, _ = actor_core_network:parameters()
        readDict.setToPretrainedEmbeddings(parameters[1])
     end



     paramxA, paramdxA = actor_core_network:getParameters()

     actorRNNs = {}

     for i=1,params.seq_length do
        actorRNNs[i] = g_clone(actor_core_network)
     end

     -- ATTENTION
     local attentionNetwork = attention.createAttentionNetwork()
     paramxRA, paramdxRA = attentionNetwork:getParameters()

     attentionNetworks = {}

     for i=1,params.seq_length do
        attentionNetworks[i] = g_clone(attentionNetwork)
     end



  elseif true then

     print("LOADING MODEL AT "..BASE_DIRECTORY.."/model-"..fileToBeLoaded)
-- TODO add params
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, SparamxRA, SparamdxRA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load(BASE_DIRECTORY.."/model-"..fileToBeLoaded, "binary"))

    if SparamxB == nil and USE_BIDIR_BASELINE and DO_TRAINING and IS_CONTINUING_ATTENTION then
        print("962 no baseline in saved file")
        crash()
    end

    print(params2)


------
     local reader_core_network
     if params.TASK == 'autoencoding' then
        reader_core_network = autoencoding.create_network(false)
     elseif params.TASK == 'combined' then
        reader_core_network = autoencoding.create_network(true,false,true)
        -- note since the output is the last parameter, one can load models that had not been 'combined'
     else
        crash()
     end


     -- LOAD PARAMETERS
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()
     for j=1, #SparamxR do
           --print(reader_core_network:parameters()[j])
           --print(SparamxR[j])
           reader_network_params[j]:set(SparamxR[j])
           reader_network_gradparams[j]:set(SparamxR[j])
     end
     paramxR, paramdxR = reader_core_network:getParameters()
     -- they are taken again so that they can be used for word embeddings a few lines further down by the attention network
     reader_network_params, reader_network_gradparams = reader_core_network:parameters()


     -- CLONE
     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network)
     end

------
     -- ACTOR
     local actor_core_network = autoencoding.create_network(true)
     actor_network_params, actor_network_gradparams = actor_core_network:parameters()
     for j=1, #SparamxA do
           actor_network_params[j]:set(SparamxA[j])
           actor_network_gradparams[j]:set(SparamdxA[j])
     end


     paramxA, paramdxA = actor_core_network:getParameters()

     actorRNNs = {}

     for i=1,params.seq_length do
        actorRNNs[i] = g_clone(actor_core_network)
     end

     -- ATTENTION

     local attentionNetwork = attention.createAttentionNetwork() --createAttentionNetwork()
     att_network_params, network_gradparams = attentionNetwork:parameters()
     if params.ATTENTION_WITH_EMBEDDINGS then
        if not IS_CONTINUING_ATTENTION then
--           print( att_network_params[1]:size())
  --         print(reader_network_params[1]:size())
           att_network_params[1]:set(reader_network_params[1])
           print("Using embeddings from the reader")
        else
           print("Not using embeddings from the reader because continuing attention")
        end
     end


     if USE_BIDIR_BASELINE and DO_TRAINING then
          setupBidirBaseline(reader_network_params, SparamxB, SparamdxB)
     end

--[[     if SparamxB ~= nil and USE_BIDIR_BASELINE and DO_TRAINING then
        print("Getting baseline network from file")
--         print(bidir_baseline_params_table)
  --       print(SparamxB)
            for j=1, #SparamxB do
               bidir_baseline_params_table[j]:set(SparamxB[j])
               bidir_baseline_gradparams_table[j]:set(SparamdxB[j])
            end
     end
]]




     if IS_CONTINUING_ATTENTION then
         network_params, network_gradparams = attentionNetwork:parameters()

         for j=1, #SparamxRA do
            network_params[j]:set(SparamxRA[j])
            network_gradparams[j]:set(SparamdxRA[j])
         end


         print("Got attention network from file")
         --attentionNetwork:getParameters():uniform(-params.init_weight, params.init_weight) -- hullu huhu
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

     print("Sequences read by model "..sentencesRead)

     reader_c[0] = readerCStart
     reader_h[0] = readerHStart
   end
end



function autoencoding.fpAutoencoding(corpus, startIndex, endIndex)


  probabilityOfChoices:fill(1)
  totalAttentions:fill(params.ATTENTION_VALUES_BASELINE) --hullu
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)

  if params.TASK == 'combined' then
     nll_reader:zero()
     reader_output = {}
  end

  --lossesForItems = {}

  for i=1, params.seq_length do
     local inputTensor = inputTensors[i]
     if USE_PREDICTION_FOR_ATTENTION then
        if i>1 then
   --        for batchItem=1,params.batch_size do
              local _, predictedTokensLM =  torch.min(reader_output[i-1],2)
            attention_inputTensors[i] = predictedTokensLM:view(-1)
--            print(predictedTokensLM)
     --      end
        else 
           attention_inputTensors[i] = inputTensors[i]
        end
     end
      --lossesForItems[i] = torch.CudaTensor(params.batch_size,1)


     -- make attention decisions

      if params.TASK == 'autoencoding' then
          local attendedInputTensor, probability = hardAttention.makeAttentionDecisions(i, inputTensor)
          reader_c[i], reader_h[i] = unpack(readerRNNs[i]:forward({attendedInputTensor, reader_c[i-1], reader_h[i-1]}))
      elseif params.TASK == 'combined' then
          if i>1 then
             surprisal_values[i] = retrieveSurprisalValue(reader_output[i-1], inputTensors[i])
          end
          if (not USE_PREDICTION_FOR_ATTENTION) and attention_inputTensors[i] ~= nil then
            crash()
          elseif PREDICTION_FOR_ATTENTION and attention_inputTensors[i] == nil then
            crash()
          end
  --        print(1729)
          local attendedInputTensor, probability = hardAttention.makeAttentionDecisions(i, inputTensor, surprisal_values[i], attention_inputTensors[i])
          reader_c[i], reader_h[i], reader_output[i] = unpack(readerRNNs[i]:forward({attendedInputTensor, reader_c[i-1], reader_h[i-1]}))
          if i < params.seq_length then

             for item=1, params.batch_size do
                local lm_loss_for_item =  reader_output[i][item][getFromData(corpus,startIndex+ item - 1,i+1)] 
                nll_reader[item] = nll_reader[item] + lm_loss_for_item-- TODO --halla
                --lossesForItems[i][item] = lm_loss_for_item
             end
          end
      end
  end

  actor_c[0] = reader_c[params.seq_length] 
  actor_h[0] = reader_h[params.seq_length] 

  --print(reader_h[20])

  nll:zero()
  actor_output = {}
  
  local inputTensor
  for i=1, params.seq_length do
     inputTensor = inputTensors[i-1]
     actor_c[i], actor_h[i], actor_output[i] = unpack(actorRNNs[i]:forward({inputTensor, actor_c[i-1], actor_h[i-1]}))
     for item=1, params.batch_size do
        local rec_loss_for_item =  actor_output[i][item][getFromData(corpus,startIndex+ item - 1,i)] 
        nll[item] = nll[item] + rec_loss_for_item -- TODO
--        lossesForItems[i][item] = lossesForItems[i][item] + rec_loss_for_item 
     end
  end






  return nll, actor_output
end



function autoencoding.bpAutoencoding(corpus, startIndex, endIndex)

  paramdxR:zero()
  paramdxA:zero()
  --paramdxRA:zero() -- will be dealt with by momentum stuff

  -- MOMENTUM
  paramdxRA:mul(params.lr_momentum / (1-params.lr_momentum))


  reset_ds()


  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)


  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)


  if params.lr > 0 or train_autoencoding then
   -- do it for actor network
      for i = params.seq_length, 1, -1 do
          inputTensor = inputTensors[i-1]
          local prior_c = actor_c[i-1]
          local prior_h = actor_h[i-1]
          local derr = transfer_data(torch.ones(1))


          local tmp = actorRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                       model.dsA)
          model.dsA[1]:copy(tmp[2])
          model.dsA[2]:copy(tmp[3])
          model.dsA[3]:zero()



          buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, i-1) -- NOTE i-1 because it is for the next round!!!
          cutorch.synchronize()
      end

      model.dsR[1]:copy(model.dsA[1])
      model.dsR[2]:copy(model.dsA[2])


      -- TODO first c, h are not trained
      -- do it for reader network
      for i = params.seq_length, 1, -1 do
    
          inputTensor= torch.cmul(inputTensors[i], attention_decisions[i])

          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
          local derr = transfer_data(torch.ones(1))
 
          local tmp = readerRNNs[i]:backward({inputTensor, prior_c, prior_h},
                                        model.dsR)
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

      momentum = 0.8

      paramxR:add(paramdxR:mul(-params.lr))
      paramxA:add(paramdxA:mul(-params.lr))

  end

  if true and train_attention_network then
     local reward = torch.add(nll, params.TOTAL_ATTENTIONS_WEIGHT,totalAttentions) -- gives the reward for each batch item
     local rewardDifference = reward:cuda():add(-rewardBaseline, ones)
     rewardBaseline = 0.8 * rewardBaseline + 0.2 * torch.sum(reward) * 1/params.batch_size
     rewardDifference:mul(REWARD_DIFFERENCE_SCALING)
     for i = params.seq_length, 1, -1 do
        local whatToMultiplyToTheFinalDerivative = torch.CudaTensor(params.batch_size)
        local attentionEntropyFactor =  torch.CudaTensor(params.batch_size)
        for j=1,params.batch_size do
          attentionEntropyFactor[j] = params.ENTROPY_PENALTY * (math.log(attention_scores[i][j][1]) - math.log(1 - attention_scores[i][j][1]))
           if attention_decisions[i][j] == 0 then
               whatToMultiplyToTheFinalDerivative[j] = -1 / (1 - attention_scores[i][j][1])
           else
               whatToMultiplyToTheFinalDerivative[j] = 1 / (attention_scores[i][j][1])
           end
        end
        local factorsForTheDerivatives =  rewardDifference:clone():cmul(whatToMultiplyToTheFinalDerivative)
        factorsForTheDerivatives:add(attentionEntropyFactor)
        local tmp = attentionNetworks[i]:backward({inputTensors[i], reader_c[i-1]},factorsForTheDerivatives)
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




