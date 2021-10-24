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
				var tokens = data.result.tokens;
				var labels = data.result.labels;
				var key = data.result.attribute;
				var offset = 0;
				var outputTokens = [];
				var markLeft = '<mark data-entity="org">';
				var markRight = '</mark>'
				var ouputValue = [];
				for (var i = 0; i < labels.length; i++){
					var value = labels[i].value;
					var position = labels[i].position;
					var start = position[0];
					var end = position[1];
					ouputValue.push(value);
					tokens.splice(start + offset, 0, markLeft);
					tokens.splice(end + 2 + offset, 0, markRight);
					offset += 2;
				}
				// remove [SE], [CLS] token, and combine sub-words
				for (var i = 0; i < tokens.length; i ++){
					if (tokens[i] == '[CLS]') {
						continue;
					}
					if ((tokens[i] == '[SEP]') || (tokens[i] == '[PAD]')) {
						break;
					}
					if (((tokens[i].substring(0, 2)) == '##') && (outputTokens.length > 0)) {

						if ((outputTokens[outputTokens.length - 1] == markLeft) || ((outputTokens[outputTokens.length - 1] == markRight))){
							outputTokens[outputTokens.length - 2] = outputTokens[outputTokens.length - 2].concat('', tokens[i].substring(2));
						} else {
							outputTokens[outputTokens.length - 1] = outputTokens[outputTokens.length - 1].concat('', tokens[i].substring(2));
						}
					}
					else {
						outputTokens.push(tokens[i])
					}

				}
				$('#successAlert').show();
				document.getElementById('outputNote').innerHTML = "<b>Note:</b> " + outputTokens.join(' ');
				document.getElementById('outputKey').innerHTML = "<b>Key:</b> " + key;
				document.getElementById('outputValue').innerHTML = "<b>Value:</b> " + ouputValue.join(' ');
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