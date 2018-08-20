require 'nn.GaussianTensor'

phono = {pars = {repertoireSize = 20,
              phonoDim = 20}}

function buildPhonoReaderModule()
  local x                = nn.Identity()() --the input, a phonological unit encoded as a one-hot vector
  local prev_c           = nn.Identity()()
  local prev_h           = nn.Identity()()
  local phoneticFeatures = nn.LookupTable(phono.pars.repertoireSize, phono.pars.phonoDim)(x) --have to initialize this somehow
  local noise = nn.GaussianTensor(phono.pars.phonoDim)()
  local confusion = nn.CAddTable()({phoneticFeatures, noise})
  local next_c, next_h = lstm.lstm(i, prev_c, prev_h)
  local module = nn.gModule({x, prev_c, prev_h},{next_c, next_h})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end



function buildPhonoReconstructorModule()
   return create_network(true)
end

function setupPhono()
  print("Creating a RNN LSTM network.")



  -- initialize data structures
  -- TODO why is there no declaration for model.sA, model.sRA here?
  model.sR = {}
  model.dsR = {}
  model.sA = {}
  model.dsA = {}

  model.start_sR = {}
  for j = 0, params.seq_length do
    model.sR[j] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  reader_c ={}
  reader_h = {}

  actor_c ={[0] = torch.CudaTensor(params.rnn_size)}
  actor_h = {[0] = torch.CudaTensor(params.rnn_size)}
  probabilities ={[0] = torch.CudaTensor(params.rnn_size)}

  reader_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero() --TODO they have to be intiialized decently
  reader_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()

  nll = torch.FloatTensor(params.batch_size)

  ones = torch.ones(params.batch_size):cuda()
  rewardBaseline = 0




   if not LOAD then
     -- READER
     local reader_core_network = buildPhonoReaderModule()
     paramxR, paramdxR = reader_core_network:getParameters()
     readerRNNs = {}
     for i=1,params.seq_length do
        readerRNNs[i] = g_clone(reader_core_network)
     end

     -- ACTOR
     local actor_core_network = buildPhonoReconstructorModule()
     paramxA, paramdxA = actor_core_network:getParameters()
     actorRNNs = {}
     for i=1,params.seq_length do
        actorRNNs[i] = g_clone(actor_core_network)
     end

  elseif true then

     print("LOADING MODEL AT ".."/disk/scratch2/s1582047/model-"..fileToBeLoaded)
-- TODO add params
     local params2, sentencesRead, SparamxR, SparamdxR, SparamxA, SparamdxA, readerCStart, readerHStart = unpack(torch.load("/disk/scratch2/s1582047/model-"..fileToBeLoaded, "binary"))

     print(params2)


     local reader_core_network
     reader_core_network = buildPhonoReaderModule()
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

     -- ACTOR
     local actor_core_network = buildPhonoReconstructorModule()
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


     -- for safety zero initialization when later using momentum
     paramdxR:zero()
     paramdxA:zero()

     print("Sequences read by model "..sentencesRead)

     reader_c[0] = readerCStart
     reader_h[0] = readerHStart
   end
end

function buildPhonoInputTensors(corpus, startIndex, endIndex)
    local inputTensors = {}
    local lengths = {}
    for token=0, phono.pars.maxInputLength do
      inputTensors[token] = torch.CudaTensor(params.batch_size)
      inputTensors[token][0] = 1
    end

      -- the batch elements
      for index=startIndex,endIndex do --the batch elements
         lengths[index] = 1
         for token = 1, params.seq_length do
              local charseq = chars[getFromData(data,index,token)]
              for i=1, #charseq do
                 local num = string.byte(charseq:sub(i,i)) - 96
                 lengths[index] = lengths[index] + 1
                 if lengths[index] > phono.pars.maxInputLength then
                     print("WARRNING 147")
                     break
                 end
                 inputTensors[lengths[index]][index] = math.min(num, 26)
              end
         end
      end
    print(inputTensors)
    return inputTensors, lengths
end

function fpPhono(corpus, startIndex, endIndex)


  local inputTensors, lengths = buildPhonoInputTensors(corpus, startIndex, endIndex)
  local targetTensors = buildInputTensors(corpus, startIndex, endIndex)

  for i=1, params.seq_length do
     inputTensor = inputTensors[i]
     reader_c[i], reader_h[i] = unpack(readerRNNs[i]:forward({inputTensors, reader_c[i-1], reader_h[i-1]}))
  end

  actor_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size)
  actor_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size)
  for i=1, params.batch_size do
    actor_c[0][i] = reader_c[lengths[i]][i] 
    actor_h[0][i] = reader_h[lengths[i]][i]
  end

  nll:zero()
  actor_output = {}
  for i=1, params.seq_length do
     inputTensor = inputTensors[i-1]
     actor_c[i], actor_h[i], actor_output[i] = unpack(actorRNNs[i]:forward({targetTensors, actor_c[i-1], actor_h[i-1]}))
     for item=1, params.batch_size do
        nll[item] = nll[item] + actor_output[i][item][getFromData(corpus,startIndex+ item - 1,i)]
     end
  end

  return nll, actor_output
end



function bpPhono(corpus, startIndex, endIndex)

  -- MOMENTUM
  paramdxA:mul(params.lr_momentum / (1-params.lr_momentum))
  paramdxR:mul(params.lr_momentum / (1-params.lr_momentum))

  reset_ds()


  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)


  local inputTensorsPhono, lengths = buildPhonoInputTensors(corpus, startIndex, endIndex)
  local targetTensors = buildInputTensors(corpus, startIndex, endIndex)



   local maxLength = 0
   for i=1, params.batch_size do
      maxLength = math.max(maxLength, lengths[i])
   end


   -- do it for actor network
      for i = params.seq_length, 1, -1 do
          inputTensor = inputTensors[i-1]
          local prior_c = actor_c[i-1]
          local prior_h = actor_h[i-1]

          local tmp = actorRNNs[i]:backward({targetTensors[i], prior_c, prior_h},
                                       model.dsA)
          model.dsA[1]:copy(tmp[2])
          model.dsA[2]:copy(tmp[3])
          model.dsA[3]:zero()

          buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, i-1) -- NOTE i-1 because it is for the next round!!!
          cutorch.synchronize()
      end

      model.dsR[1]:zero()
      model.dsR[2]:zero()



      -- TODO first c, h are not trained
      -- do it for reader network
      for i = maxLength, 1, -1 do
          for j=1, params.batch_size do
              if lengths[j] == i then
                  model.dsR[1][j] = model.dsA[1][j]
                  model.dsR[2][j] = model.dsA[2][j]
              end
          end

          local prior_c = reader_c[i-1]
          local prior_h = reader_h[i-1]
 
          local tmp = readerRNNs[i]:backward({inputTensorsPhono[i], prior_c, prior_h},
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

     paramdxA:mul((1-params.lr_momentum))
     paramxA:add(paramdxA:mul(- 1 * params.lr_att))
     paramdxA:mul(1 / (- 1 * params.lr_att)) -- is this really better than cloning before multiplying?

     paramdxR:mul((1-params.lr_momentum))
     paramxR:add(paramdxR:mul(- 1 * params.lr_att))
     paramdxR:mul(1 / (- 1 * params.lr_att)) -- is this really better than cloning before multiplying?

     assert(norm_dwR == norm_dwR)
     assert(norm_dwA == norm_dwA)


end



function printStuffForPhono(perp, actor_output, since_beginning, epoch, numberOfWords)
            print("+++++++ "..perp[1]..'  '..meanNLL)
             print(epoch.."  "..corpusReading.currentFile..
               '   since beginning = ' .. since_beginning .. ' mins.')
            print(experimentNameOut)
            print(torch.sum(nll)/params.batch_size)
            --print(nll:add(-1,nll_reader))

            print(params)

            for l = 1, 1 do
               print("....")
               print(perp[l])
               for j=1,params.seq_length do
                  local predictedScores, predictedTokens = torch.min(actor_output[j][l],1)
                  local predictedScoresLM, predictedTokensLM = torch.min(reader_output[j-1][l],1)
                  io.write((chars[getFromData(corpus,l,j)]))--..'\n')
                  io.write(" \t "..chars[predictedTokens[1]].."  "..math.exp(-predictedScores[1]).."  "..math.exp(-actor_output[j][l][getFromData(corpus,l,j)]).." \t "..chars[predictedTokensLM[1]].."  "..math.exp(-predictedScoresLM[1]).."\n")
               end
            end
            --io.output(fileStats)
            fileStats:write((numberOfWords/params.seq_length)..'\t'..perp[1]..'\n')
            fileStats:flush()
            --io.output(stdout)


end


function savePhono(numberOfWords)
           local uR, udR = readerRNNs[1]:parameters()
           local uA, udA = actorRNNs[1]:parameters()
           modelsArray = {params,(numberOfWords/params.seq_length),uR, udR, uA, udA, reader_c[0], reader_h[0]}
end



