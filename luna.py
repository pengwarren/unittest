from smartroom import Smartroom

luna = Smartroom()
# for luna.text in luna.configurations["TEST_DATA"]:
for luna.text in ["please don't turn on the lights"]:
    try:
        print(luna.text)
        if luna.response:
            luna.throw_parameter_exception()
        luna.perform_classification()
        print(luna)
        luna.perform_naive_bayes_classification()
        print(luna)
        response = luna.perform_request(luna.POST, "login", luna.credentials)
        for parameter, (action, state) in luna.response.items():
            luna.telegram = False if state == 0 else True
            universal, lights, printer, tv = luna.configurations["PARAMETERS"]
            if parameter in lights:
                luna.perform_request(
                    luna.PUTS,
                    "datapoints/1",
                    luna.telegram,
                    dict(user=response.text)
                )
                luna.perform_request(
                    luna.PUTS,
                    "datapoints/2",
                    luna.telegram,
                    dict(user=response.text)
                )
            elif parameter in tv:
                luna.perform_request(
                    luna.PUTS,
                    "datapoints/3",
                    luna.telegram,
                    dict(user=response.text)
                )
            elif parameter in printer:
                luna.perform_request(
                    luna.PUTS,
                    "datapoints/4",
                    luna.telegram,
                    dict(user=response.text)
                )
            elif parameter in universal:
                luna.perform_request(
                    luna.PUTS,
                    "datapoints/1",
                    luna.telegram,
                    dict(user=response.text)
                )
                luna.perform_request(
                    luna.PUTS,
                    "datapoints/2",
                    luna.telegram,
                    dict(user=response.text)
                )
                luna.perform_request(
                        luna.PUTS,
                        "datapoints/3",
                        luna.telegram,
                        dict(user=response.text)
                )
                luna.perform_request(
                        luna.PUTS,
                        "datapoints/4",
                        luna.telegram,
                        dict(user=response.text)
                )
            else:
                raise NotImplementedError
    except Smartroom.ParameterError as e:
        print(e)
        continue
    except NotImplementedError:
        continue
luna.state = luna.BACK
del luna
