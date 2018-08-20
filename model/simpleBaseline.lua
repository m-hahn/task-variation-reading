simpleBaseline = {}
simpleBaseline.__name = "simpleBaseline.lua"
print(simpleBaseline)




function createSimpleBaseline()
  local model = nn.Sequential()
  model:add(nn.View(-1,1))
  model:add(nn.Linear(1,1))

--[[  model:add(nn.SplitTable(1,1))

  local parallel = nn.ParallelTable()
  parallel:add(nn.Reshape(1))
  parallel:add(nn.Reshape(1))

  model:add(parallel)
  model:add(nn.SelectTable(2))]]



--  model:add(nn.LSTM(2, params.baseline_rnn_size))




  --print(model:forward(testTensor))
--  print(model:forward(testTensor)[1]) 

    return model:cuda()


end

function setupSimpleBaseline(SparamxB, SparamdxB)
        simpleBaseline.criterion_gradients = torch.CudaTensor(params.batch_size,1)

        simple_baseline = createSimpleBaseline()


        local simple_baseline_params_table, simple_baseline_gradparams_table = simple_baseline:parameters()

     if SparamxB ~= nil and #SparamxB == #simple_baseline_params_table then
        print("Getting baseline network from file")
        for j=1, #SparamxB do
            simple_baseline_params_table[j]:set(SparamxB[j])
             simple_baseline_gradparams_table[j]:set(SparamdxB[j])
         end
     end

       simple_baseline_params, simple_baseline_gradparams = simple_baseline:getParameters()


       simple_baseline_params_table, simple_baseline_gradparams_table = nil

        baseline_criterion = nn.MSECriterion():cuda()
        
end

