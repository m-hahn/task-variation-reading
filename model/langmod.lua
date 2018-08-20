langmod = {}
assert(false)



function setupLanguageModelling()
  print("Creating a RNN LSTM network.")



  -- initialize data structures

  model.dsA = {}

  model.dsA[1] = transfer_data(torch.zeros(params.rnn_size))
  model.dsA[2] = transfer_data(torch.zeros(params.rnn_size))
  model.dsA[3] = transfer_data(torch.zeros(params.rnn_size)) -- NOTE actually will later have different size



  actor_c ={[0] = torch.CudaTensor(params.rnn_size)}
  actor_h = {[0] = torch.CudaTensor(params.rnn_size)}

  actor_c[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero() --TODO they have to be intiialized decently
  actor_h[0] = torch.CudaTensor(params.batch_size,params.rnn_size):zero()

   nll = torch.FloatTensor(params.batch_size)



  ones = torch.ones(params.batch_size):cuda()


   if not LOAD then
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
     local params2, sentencesRead, SparamxA, SparamdxA, actorCStart, actorHStart = unpack(torch.load("/disk/scratch2/s1582047/model-"..fileToBeLoaded, "binary"))

    print(params2)


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

     actor_c[0] = actorCStart
     actor_h[0] = actorHStart


   end
end



function fpLanguageModelling(corpus, startIndex, endIndex)
  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)

  nll:zero()
  actor_output = {}
  for i=1, params.seq_length do
     inputTensor = inputTensors[i-1]
     actor_c[i], actor_h[i], actor_output[i] = unpack(actorRNNs[i]:forward({inputTensor, actor_c[i-1], actor_h[i-1]}))
     for item=1, params.batch_size do
        nll[item] = nll[item] + actor_output[i][item][getFromData(corpus,startIndex+ item - 1,i)] -- TODO
     end
  end

  return nll, actor_output
end





function bpLanguageModelling(corpus, startIndex, endIndex)
  paramdxA:zero()

  reset_ds()

  buildGradientsOfProbOutputs(model.dsA[3], corpus, startIndex, endIndex, params.seq_length)


  local inputTensors = buildInputTensors(corpus, startIndex, endIndex)


  if train_autoencoding then
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

      model.norm_dwA = paramdxA:norm()
      if model.norm_dwA > params.max_grad_norm then
          local shrink_factor = params.max_grad_norm / model.norm_dwA
          paramdxA:mul(shrink_factor)
      end


      paramxA:add(paramdxA:mul(-params.lr))
  else
     crash()
  end

end




function printStuffForLangmod(perp, actor_output, since_beginning, epoch, numberOfWords)

            print("+++++++ "..perp[1]..'  '..meanNLL)
             print(epoch.."  "..readChunks.corpusReading.currentFile..
               '   since beginning = ' .. since_beginning .. ' mins.')  
            print(experimentNameOut)
            print(params) 
   
            for l = 1, 1 do
               print("....")
               print(perp[l])
               for j=1,params.seq_length do
                  local predictedScores, predictedTokens = torch.min(actor_output[j][l],1)
                  io.write((readDict.chars[getFromData(corpus,l,j)]))--..'\n')
                  io.write(" ~ "..readDict.chars[predictedTokens[1]].."  "..math.exp(-predictedScores[1]).."  "..math.exp(-actor_output[j][l][getFromData(corpus,l,j)]).."\n")
               end
            end
            --io.output(fileStats)
            fileStats:write((numberOfWords/params.seq_length)..'\t'..perp[1]..'\n')
            fileStats:flush()
            --io.output(stdout)


end


