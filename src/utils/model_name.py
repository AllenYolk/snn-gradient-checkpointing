class ModelNameGenerator:

    available_projs = [
        "cifar10-online",
        "cifar100-online",
        "cifar10dvs-online",
        "sequential-cifar10-me",
        "sequential-cifar100-me",
        "gsc-online",
        "shd-online",
        "ssc-online",
        "dvslip-online",
        "ncars-online",
        "gen1-online",
        "solar-forecast-online",
        "electricity-forecast-online",
        "metr-la-forecast-online",
        "pems-bay-forecast-online",
    ]

    def __init__(self, proj):
        self.proj = proj

    @staticmethod
    def generate_cifar10_model_name(args):
        run_name = f"{args.model}_{args.learning_rule}_T{args.T}_"

        if args.model.startswith("qkformer"):
            run_name += (
                f"decaylambda{args.decay_lambda}_optAdamForQKFormer_"
                f"lr{args.learning_rate}_optCosALRForQKFormer_"
            )
        else:
            if args.lr_scheduler == 'CosALR':
                sch_str = f'optCosALR_Tmax{args.T_max}'
            elif args.lr_scheduler == 'StepLR':
                sch_str = (
                    f'optStepLR_stepsize{args.step_size}_gamma{args.gamma}'
                )
            else:
                raise NotImplementedError(args.lr_scheduler)
            run_name += (
                f"{args.model}_{args.learning_rule}_T{args.T}_"
                f"decaylambda{args.decay_lambda}_opt{args.optimizer}_"
                f"lr{args.learning_rate}_{sch_str}_"
            )

        run_name += (
            f"amp{args.amp}_ou{args.online_update}_"
            f"bn{args.batch_normalization}_"
            f"ots{args.online_threshold_stabilization}_"
            f"l2factor{args.l2_factor}_losslambda{args.loss_lambda}_"
            f"rtnl{args.rate_till_now_loss}_"
            f"ow{args.optimizer_wrapper}_owr{args.optimizer_wrapper_rho}_"
            f"br{args.block_rho}"
        )
        return run_name

    @staticmethod
    def generate_cifar100_model_name(args):
        return ModelNameGenerator.generate_cifar10_model_name(args)

    @staticmethod
    def generate_cifar10dvs_model_name(args):
        return (
            ModelNameGenerator.generate_cifar10_model_name(args) +
            f"_dr{args.dropout}"
        )

    @staticmethod
    def generate_sequential_cifar10_model_name(args):
        run_name = (
            f"c{args.channels}_"
            f"decaylambda{args.decay_lambda}_opt{args.optimizer}_"
            f"lr{args.learning_rate}_amp{args.amp}_"
        )
        return run_name

    @staticmethod
    def generate_sequential_cifar100_model_name(args):
        return ModelNameGenerator.generate_sequential_cifar10_model_name(args)

    @staticmethod
    def generate_gsc_model_name(args):
        run_name = (
            f"{args.learning_rule}_c{args.channels}_"
            f"decaylambda{args.decay_lambda}_opt{args.optimizer}_"
            f"lr{args.learning_rate}_gamma{args.gamma}_"
            f"l2factor{args.l2_factor}_amp{args.amp}_"
            f"ou{args.online_update}_ws{args.weight_standardization}_"
            f"bn{args.batch_normalization}_"
            f"ots{args.online_threshold_stabilization}_"
            f"losslambda{args.loss_lambda}_"
            f"rtnl{args.rate_till_now_loss}_"
            f"oorl{args.offline_overall_rate_loss}_"
            f"ow{args.optimizer_wrapper}_owr{args.optimizer_wrapper_rho}_"
            f"br{args.block_rho}"
        )
        return run_name

    @staticmethod
    def generate_shd_model_name(args):
        run_name = (
            f"{args.learning_rule}_au{args.enable_augmentation}_"
            f"m{args.model}_c{args.channels}_"
            f"decaylambda{args.decay_lambda}_opt{args.optimizer}_"
            f"lr{args.learning_rate}_gamma{args.gamma}_"
            f"l2factor{args.l2_factor}_amp{args.amp}_"
            f"ou{args.online_update}_ws{args.weight_standardization}_"
            f"bn{args.batch_normalization}_"
            f"ots{args.online_threshold_stabilization}_"
            f"losslambda{args.loss_lambda}_"
            f"rtnl{args.rate_till_now_loss}_"
            f"oorl{args.offline_overall_rate_loss}_"
            f"ow{args.optimizer_wrapper}_owr{args.optimizer_wrapper_rho}_"
            f"br{args.block_rho}"
        )
        return run_name

    @staticmethod
    def generate_ssc_model_name(args):
        return ModelNameGenerator.generate_shd_model_name(args)

    @staticmethod
    def generate_dvslip_model_name(args):
        run_name = (
            f"{args.learning_rule}_T{args.T}_"
            f"decaylambda{args.decay_lambda}_plif{args.use_plif}_"
            f"ou{args.online_update}_ws{args.weight_standardization}_"
            f"bn{args.batch_normalization}_"
            f"ots{args.online_threshold_stabilization}_"
            f"losslambda{args.loss_lambda}_"
            f"rtnl{args.rate_till_now_loss}_"
            f"oorl{args.offline_overall_rate_loss}_"
            f"ow{args.optimizer_wrapper}_owr{args.optimizer_wrapper_rho}_"
            f"br{args.block_rho}"
        )
        return run_name

    @staticmethod
    def generate_ncars_model_name(args):
        run_name = (
            f"{args.model}_{args.learning_rule}_T{args.T}_"
            f"lr{args.learning_rate}_ou{args.online_update}_"
            f"ws{args.weight_standardization}_bn{args.batch_normalization}_"
            f"ots{args.online_threshold_stabilization}_"
            f"losslambda{args.loss_lambda}_"
            f"rtnl{args.rate_till_now_loss}_"
            f"oorl{args.offline_overall_rate_loss}_"
            f"ow{args.optimizer_wrapper}_owr{args.optimizer_wrapper_rho}_"
            f"br{args.block_rho}"
        )
        return run_name

    @staticmethod
    def generate_gen1_model_name(args):
        run_name = (
            f"{args.model}_{args.learning_rule}_T{args.T}_"
            f"lr{args.learning_rate}_l2factor{args.l2_factor}_"
            f"ou{args.online_update}_"
            f"ws{args.weight_standardization}_bn{args.batch_normalization}_"
            f"ots{args.online_threshold_stabilization}_"
            f"rtnl{args.rate_till_now_loss}_"
            f"oorl{args.offline_overall_rate_loss}_"
            f"ow{args.optimizer_wrapper}_owr{args.optimizer_wrapper_rho}_"
            f"br{args.block_rho}"
        )
        return run_name

    @staticmethod
    def generate_forecast_model_name(args):
        run_name = (
            f"{args.task}_{args.learning_rule}_"
            f"dn{args.data_normalization}_"
            f"gse{args.global_spike_encoder}_"
            f"T{args.T}_H{args.horizon}_"
            f"h{args.hidden_size}_l{args.layers}_"
            f"decaylambda{args.decay_lambda}_"
            f"lr{args.learning_rate}_amp{args.amp}_"
            f"ou{args.online_update}_ws{args.weight_standardization}_"
            f"bn{args.batch_normalization}_"
            f"ots{args.online_threshold_stabilization}_"
            f"rtnl{args.rate_till_now_loss}_"
            f"oorl{args.offline_overall_rate_loss}_"
            f"ow{args.optimizer_wrapper}_owr{args.optimizer_wrapper_rho}_"
            f"br{args.block_rho}"
        )
        return run_name

    def generate(self, args):
        if self.proj == "cifar10-online":
            return self.generate_cifar10_model_name(args)
        elif self.proj == "cifar100-online":
            return self.generate_cifar100_model_name(args)
        elif self.proj == "cifar10dvs-online":
            return self.generate_cifar10dvs_model_name(args)
        elif self.proj == "sequential-cifar10-me":
            return self.generate_sequential_cifar10_model_name(args)
        elif self.proj == "sequential-cifar100-me":
            return self.generate_sequential_cifar100_model_name(args)
        elif self.proj == "gsc-online":
            return self.generate_gsc_model_name(args)
        elif self.proj == "shd-online":
            return self.generate_shd_model_name(args)
        elif self.proj == "ssc-online":
            return self.generate_ssc_model_name(args)
        elif self.proj == "dvslip-online":
            return self.generate_dvslip_model_name(args)
        elif self.proj == "ncars-online":
            return self.generate_ncars_model_name(args)
        elif self.proj == "gen1-online":
            return self.generate_gen1_model_name(args)
        elif self.proj == "solar-forecast-online":
            return self.generate_forecast_model_name(args)
        elif self.proj == "electricity-forecast-online":
            return self.generate_forecast_model_name(args)
        elif self.proj == "metr-la-forecast-online":
            return self.generate_forecast_model_name(args)
        elif self.proj == "pems-bay-forecast-online":
            return self.generate_forecast_model_name(args)
        else:
            raise ValueError(
                f"Unknown project: {self.proj}; "
                f"available projects: {self.available_projs}"
            )


class ModelNameParser:

    available_projs = [
        "cifar10-online",
        "cifar100-online",
        "cifar10dvs-online",
        "sequential-cifar10-online",
        "sequential-cifar100-online",
        "gsc-online",
        "shd-online",
        "ssc-online",
        "dvslip-online",
        "ncars-online",
        "gen1-online",
        "solar-forecast-online",
        "electricity-forecast-online",
        "metr-la-forecast-online",
        "pems-bay-forecast-online",
    ]

    def __init__(self, proj):
        self.proj = proj

    @staticmethod
    def parse_cifar10_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_cifar100_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_cifar10dvs_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_sequential_cifar10_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_sequential_cifar100_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_gsc_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_shd_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_ssc_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_dvslip_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_ncars_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_gen1_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_forecast_model_name(model_name):
        raise NotImplementedError

    def parse(self, model_name):
        if self.proj == "cifar10-online":
            return self.parse_cifar10_model_name(model_name)
        elif self.proj == "cifar100-online":
            return self.parse_cifar100_model_name(model_name)
        elif self.proj == "cifar10dvs-online":
            return self.parse_cifar10dvs_model_name(model_name)
        elif self.proj == "sequential-cifar10-online":
            return self.parse_sequential_cifar10_model_name(model_name)
        elif self.proj == "sequential-cifar100-online":
            return self.parse_sequential_cifar100_model_name(model_name)
        elif self.proj == "gsc-online":
            return self.parse_gsc_model_name(model_name)
        elif self.proj == "shd-online":
            return self.parse_shd_model_name(model_name)
        elif self.proj == "ssc-online":
            return self.parse_ssc_model_name(model_name)
        elif self.proj == "dvslip-online":
            return self.parse_dvslip_model_name(model_name)
        elif self.proj == "ncars-online":
            return self.parse_ncars_model_name(model_name)
        elif self.proj == "gen1-online":
            return self.parse_gen1_model_name(model_name)
        elif self.proj == "solar-forecast-online":
            return self.parse_forecast_model_name(model_name)
        elif self.proj == "electricity-forecast-online":
            return self.parse_forecast_model_name(model_name)
        elif self.proj == "metr-la-forecast-online":
            return self.parse_forecast_model_name(model_name)
        elif self.proj == "pems-bay-forecast-online":
            return self.parse_forecast_model_name(model_name)
        else:
            raise ValueError(
                f"Unknown project: {self.proj}; "
                f"available projects: {self.available_projs}"
            )
