$(document).ready(function() {

	$('form').on('submit', function(event) {

		$.ajax({
			data : {
				context : $('#noteInput').val(),
				attribute : $('#attributeInput').val()
			},
			type : 'POST',
			url : '/process'
		})
		.done(function(data) {
			if (data.errcode == 1) {
				var predictions = data.result.predictions;
				var key = data.result.attribute;
				var outputValue = [];
				var outputText = [];
				for (var i = 0; i < predictions.length; i++) {
					var tokens = predictions[i].tokens;
					var labels = predictions[i].labels;
					var offset = 0;
					var outputTokens = [];
					var markLeft = '<mark data-entity="weight">';
					var markRight = '</mark>'
					for (var j = 0; j < labels.length; j++){
						var value = labels[j].value;
						var position = labels[j].position;
						var start = position[0];
						var end = position[1];
						if ((value == '[SEP]') || (value == '[CLS]')) {  // special case
							continue;
						}
						outputValue.push(value);

						tokens.splice(start + offset, 0, markLeft);
						tokens.splice(end + 2 + offset, 0, markRight);
						offset += 2;
					}

					// remove [SE], [CLS] token, and combine sub-words
					for (var j = 0; j < tokens.length; j++){
						if (tokens[j] == '[CLS]') {
							continue;
						}
						if ((tokens[j] == '[SEP]') || (tokens[j] == '[PAD]')) {
							break;
						}
						if (((tokens[j].substring(0, 2)) == '##') && (outputTokens.length > 0)) {

							if ((outputTokens[outputTokens.length - 1] == markLeft) || ((outputTokens[outputTokens.length - 1] == markRight))){
								outputTokens[outputTokens.length - 2] = outputTokens[outputTokens.length - 2].concat('', tokens[j].substring(2));
							} else {
								outputTokens[outputTokens.length - 1] = outputTokens[outputTokens.length - 1].concat('', tokens[j].substring(2));
							}
						}
						else {
							outputTokens.push(tokens[j])
						}
					}
					outputText.push(outputTokens.join(' '));
				}

				$('#successAlert').show();
				document.getElementById('outputNote').innerHTML = "<b>Note:</b> " + outputText.join('. ');
				document.getElementById('outputKey').innerHTML = "<b>Key:</b> " + key;
				document.getElementById('outputValue').innerHTML = "<b>Value:</b> " + outputValue.join(', ');
				$('#errorAlert').hide();
			}
			else {
				$('#errorAlert').text("Model Failed!").show();
				$('#successAlert').hide();
			}

		});
		event.preventDefault();
	});

});