module @Lumina {
  func.func @main(%arg0: !lumina.lm_tensor<128x128x24xf32,0>) -> !lumina.lm_tensor<128x128x24xf32,0> attributes {dp_attr = #lumina.DP<DP = 2 : 0, 1>} {
    %0 = "lumina.softmax"(%arg0) <{axis = 1 : i64}> : (!lumina.lm_tensor<128x128x24xf32,0>) -> !lumina.lm_tensor<128x128x24xf32,0>
    %1 = "lumina.softmax"(%0) <{axis = 1 : i64}> : (!lumina.lm_tensor<128x128x24xf32,0>) -> !lumina.lm_tensor<128x128x24xf32,0>
    return %1 : !lumina.lm_tensor<128x128x24xf32,0>
  }
}