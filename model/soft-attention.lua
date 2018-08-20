require('nn.UniformLayer')
require('nn.ScalarMult')



local function createSoftAttentionReaderModule()
   if not (params.TASK == 'combined-soft') then
      crash()
   end
   local word_input = nn.Identity()()
   local xemb = nn.Normalize(2)(LookupTable(params.vocab_size,params.embeddings_dimensionality)(word_input))
   local prev_h = nn.Identity()()
   local prev_c = nn.Identity()()

   local inputNodes = {word_input, prev_c, prev_h}
   local attentionInputTable = {}
   local x2h
   local y2h
   local z2h
   if not attention.ABLATE_INPUT then
       x2h = nn.Linear(params.embeddings_dimensionality, params.rnn_size)(xemb)
       table.insert(attentionInputTable, x2h)
   end
   if not attention.ABLATE_STATE then
       y2h = nn.Linear(params.rnn_size, params.rnn_size)(prev_c)
       table.insert(attentionInputTable, y2h)
   end
   if not attention.ABLATE_SURPRISAL then
       local surprisal = nn.Identity()()
       z2h = nn.Linear(1, params.rnn_size)(surprisal)
       table.insert(attentionInputTable, z2h)
       table.insert(inputNodes, surprisal)
   end
   local hidden_attention
   if #attentionInputTable > 1 then
      hidden_attention = nn.Sigmoid()(nn.CAddTable()(attentionInputTable))
   else
      hidden_attention = nn.Sigmoid()(attentionInputTable[1])
   end

   local latentStatesWeight
   if params.TOTAL_ATTENTIONS_WEIGHT > 0 then
      print("Doing attention training, so setting weight to 0.2")
      latentStatesWeight = 0.2
   else
      print("Not doing attention training, so setting weight to 1.0")
      latentStatesWeight = 1.0
   end
   local attention = nn.Sigmoid()(nn.MulConstant(latentStatesWeight)((nn.Linear(params.rnn_size, 1)(hidden_attention))))
   local randomTensor = nn.Normalize(2)(nn.UniformLayer(params.batch_size , params.embeddings_dimensionality)(nn.BlockGradientLayer()(word_input)))
   local unnormalized_input = nn.CAddTable()({
                                              nn.ScalarMult()({attention,xemb}),
                                              nn.ScalarMult()({nn.MulConstant(-1)(nn.AddConstant(-1)(attention)), randomTensor})
                                             })
   -- normalizes innermost dimension
--   local normalized_input = nn.Normalize(2)(unnormalized_input)
   local next_c, next_h = lstm.lstm(unnormalized_input, prev_c, prev_h, params.embeddings_dimensionality)
   local h2y              = nn.Linear(params.rnn_size, params.vocab_size)(next_c)
   local output           = nn.MulConstant(-1)(nn.LogSoftMax()(h2y))
   local module = nn.gModule(inputNodes,
                                      {next_c, next_h, output, attention})
   module:getParameters():uniform(-params.init_weight, params.init_weight)
   return transfer_data(module)
end
-- other objective for economy




function setupCombinedSoft()
  print("Creating a RNN LSTM network.")

  if params.TASK ~= 'combined-soft' then
     crash()
  end
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
  model.dsR[3] = transfer_data(torch.zeros(params.vocab_size))
  model.dsR[4] = transfer_data(torch.zeros(params.batch_size))

  model.dsA[1] = transfer_data(torch.zeros(params.rnn_size))
  model.dsA[2] = transfer_data(torch.zeros(params.rnn_size))
 model.dsA[3] = transfer_data(torch.zeros(params.rnn_size)) -- NOTE actually will later have different size



  reader_c ={}
  reader_h = {}

  actor_c ={[0] = torch.CudaTensor(params.rnn_size)}
  actor_h = {[0] = torch.CudaTensor(params.rnn_size)}

  reader_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero() --TODO they have to be intiialized decently
  reader_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()


  reader_output = {}
  surprisal_values = {[1] = transfer_data(torch.zeros(params.batch_size,1))}



--  attention_decisions = {}
  attention_scores = {}
  baseline_scores = {}
  for i=1, params.seq_length do
  --   attention_decisions[i] = torch.CudaTensor(params.batch_size)
     attention_scores[i] = torch.CudaTensor(params.batch_size,1)
     baseline_scores[i] = torch.CudaTensor(params.batch_size,1)
  end

  totalAttentions = torch.FloatTensor(params.batch_size) -- apparently using CudaTensor would cause a noticeable slowdown...?!
  nll = torch.FloatTensor(params.batch_size)

  if params.TASK == 'combined-soft' then
      nll_reader = torch.FloatTensor(params.batch_size)
  end





  ones = transfer_parameters(torch.ones(params.batch_size))
  rewardBaseline = 0

   variance_average = 100
   recurrent_variance_average = 100
   if not LOAD then
     -- READER
     local reader_core_network
     reader_core_network = createSoftAttentionReaderModule()
     paramxR, paramdxR = reader_core_network:getParameters()

     readerRNNs = {}

     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network)
     end

     -- ACTOR
     local actor_core_network = autoencoding.create_network(true)
     paramxA, paramdxA = actor_core_network:getParameters()

     actorRNNs = {}

     for i=1,params.seq_length do
        actorRNNs[i] = g_clone(actor_core_network)
     end

  elseif true then

     print("LOADING MODEL AT ".."/disk/scratch2/s1582047/model-"..fileToBeLoaded)
-- TODO add params
     
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, readerCStart, readerHStart, SparamxB, SparamdxB = unpack(torch.load("/disk/scratch2/s1582047/model-"..fileToBeLoaded, "binary"))



----      modelsArray = {params,(numberOfWords/params.seq_length),uR, udR, uA, udA, reader_c[0], reader_h[0]}


    if SparamxB == nil and USE_BIDIR_BASELINE and DO_TRAINING and IS_CONTINUING_ATTENTION then
        print("962 no baseline in saved file")
        crash()
    end

    print(params2)


------
     local reader_core_network
     reader_core_network = createSoftAttentionReaderModule()

     local reader_network_params, reader_network_gradparams = reader_core_network:parameters()
     for j=1, #SparamxR do
           reader_network_params[j]:set(SparamxR[j])
           reader_network_gradparams[j]:set(SparamxR[j])
     end



     paramxR, paramdxR = reader_core_network:getParameters()

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


     print("Sequences read by model "..sentencesRead)

     reader_c[0] = readerCStart
     reader_h[0] = readerHStart
   end
end





function fpCombinedSoft(corpus, startIndex, endIndex)


  totalAttentions:fill(params.ATTENTION_VALUES_BASELINE) --hullu
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)

  if params.TASK == 'combined-soft' then
     nll_reader:zero()
     reader_output = {}
  end


  for i=1, params.seq_length do
     local inputTensor = inputTensors[i]
          if i>1 then
             surprisal_values[i] = retrieveSurprisalValue(reader_output[i-1], inputTensors[i])
          end

--   local module = nn.gModule({word_input, prev_c, prev_h, surprisal},
  --                                    {next_c, next_h, output, attention})

--[[print(reader_c[i-1])
print(reader_h[i-1])
print(i)
print(surprisal_values)
print(inputTensor)]]
          local inputs = {inputTensor, reader_c[i-1], reader_h[i-1]}
          if not ABLATE_SURPRISAL then
            table.insert(inputs, surprisal_values[i])
          end
          reader_c[i], reader_h[i], reader_output[i], attention_scores[i] = unpack(readerRNNs[i]:forward(inputs))
          if i < params.seq_length then
             for item=1, params.batch_size do
                local lm_loss_for_item =  reader_output[i][item][getFromData(corpus,startIndex+ item - 1,i+1)] 
                nll_reader[item] = nll_reader[item] + lm_loss_for_item-- TODO --halla
                --lossesForItems[i][item] = lm_loss_for_item
             end
          end
  end

  actor_c[0] = reader_c[params.seq_length] 
  actor_h[0] = reader_h[params.seq_length] 


  nll:zero()
  actor_output = {}
  
  local inputTensor
  for i=1, params.seq_length do
     inputTensor = inputTensors[i-1]
     actor_c[i], actor_h[i], actor_output[i] = unpack(actorRNNs[i]:forward({inputTensor, actor_c[i-1], actor_h[i-1]}))
     for item=1, params.batch_size do
        local rec_loss_for_item =  actor_output[i][item][getFromData(corpus,startIndex+ item - 1,i)] 
        nll[item] = nll[item] + rec_loss_for_item -- TODO
     end
  end

  return nll, actor_output
end




function bpCombinedSoft(corpus, startIndex, endIndex)
  
  
  paramdxR:zero()
  paramdxA:zero()
  
  
  
  reset_ds()
  
  
  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)
  
  
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)
  
  TRAIN_LANGMOD = true
  TRAIN_AUTOENCODER = true
  
  if params.lr > 0 then --hrhr
   -- do it for actor network
    if TRAIN_AUTOENCODER then
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
      model.dsR[3]:zero()
   else
      model.dsR[1]:zero()
      model.dsR[2]:zero()
      model.dsR[3]:zero()
   end
   model.dsR[4]:fill(params.TOTAL_ATTENTIONS_WEIGHT)
 

      -- TODO first c, h are not trained
      -- do it for reader network
      for i = params.seq_length, 1, -1 do
     
          inputTensor= inputTensors[i]
  
          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
          local derr = transfer_data(torch.ones(1))
 
--reader_c[i], reader_h[i], reader_output[i], attention_scores[i] = unpack(readerRNNs[i]:forward({inputTensor, reader_c[i-1], reader_h[i-1], surprisal_values[i]}))
          local inputs = {inputTensor, reader_c[i-1], reader_h[i-1]}
          if not ABLATE_SURPRISAL then
            table.insert(inputs, surprisal_values[i])
          end
 
 
          local tmp = readerRNNs[i]:backward(inputs, model.dsR)
--dsR:
--1 reader_c
--2 reader_h
--3 reader_output[i]
--4 attention_scores[i]

   --print(model.dsR[1])
   --print(model.dsR[2])
          model.dsR[1]:copy(tmp[2])
          model.dsR[2]:copy(tmp[3])
          model.dsR[3]:zero()
          if TRAIN_LANGMOD then
             buildGradientsOfProbOutputs(model.dsR[3], corpus, startIndex, endIndex, i)
          end
--          model.dsR[4]:fill(params.TOTAL_ATTENTIONS_WEIGHT)


          cutorch.synchronize()
          --print(paramdxR:norm())
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
 
      paramxR:add(paramdxR:mul(-params.lr))
      paramxA:add(paramdxA:mul(-params.lr))
      print(paramdxR:mul(-params.lr):mean()) 
  end
  
  
  
end




function printStuffForCombinedSoft(perp, actor_output, since_beginning, epoch, numberOfWords)
            print("+++++++ "..perp[1]..'  '..meanNLL)
             print(epoch.."  "..corpusReading.currentFile..
               '   since beginning = ' .. since_beginning .. ' mins.')  
            print(experimentNameOut)
            print(torch.sum(nll_reader)/params.batch_size)
            print(torch.sum(nll)/params.batch_size)
            --print(nll:add(-1,nll_reader))
            
            print(params) 
   
            for l = 1, 1 do
               print("....")
               print(perp[l])
               for j=2,params.seq_length do
                  local predictedScores, predictedTokens = torch.min(actor_output[j][l],1)
                  local predictedScoresLM, predictedTokensLM = torch.min(reader_output[j-1][l],1)
                  io.write((chars[getFromData(corpus,l,j)]))--..'\n')
                  io.write(" \t "..chars[predictedTokens[1]].."  "..math.exp(-predictedScores[1]).."  "..math.exp(-actor_output[j][l][getFromData(corpus,l,j)]).." \t "..chars[predictedTokensLM[1]].."  "..math.exp(-predictedScoresLM[1]).."  "..math.exp(-reader_output[j-1][l][getFromData(corpus,l,j)]).." \t "..attention_scores[j][l][1].."\n")
               end
            end
            --io.output(fileStats)
            fileStats:write((numberOfWords/params.seq_length)..'\t'..perp[1]..'\n')
            fileStats:flush()
            --io.output(stdout)
end


