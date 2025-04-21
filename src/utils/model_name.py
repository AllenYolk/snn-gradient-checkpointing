class ModelNameGenerator:

    available_projs = [
        "sequential-cifar10-me",
        "sequential-cifar100-me",
        "cifar10dvs-me",
    ]

    def __init__(self, proj):
        self.proj = proj

    @staticmethod
    def generate_sequential_cifar10_model_name(args):
        run_name = (
            f"{args.neuron_type}_{args.network}_{args.spike_compressor}_"
            f"lomo{args.lomo}_amp{args.amp}"
        )
        return run_name

    @staticmethod
    def generate_sequential_cifar100_model_name(args):
        return ModelNameGenerator.generate_sequential_cifar10_model_name(args)

    @staticmethod
    def generate_cifar10dvs_model_name(args):
        return (
            f"{args.neuron_type}_{args.network}_{args.spike_compressor}_"
            f"T{args.T}_lomo{args.lomo}_amp{args.amp}_loss{args.loss}_"
            f"tebn{args.allow_tebn}"
        )

    def generate(self, args):
        if self.proj == "sequential-cifar10-me":
            return self.generate_sequential_cifar10_model_name(args)
        elif self.proj == "sequential-cifar100-me":
            return self.generate_sequential_cifar100_model_name(args)
        elif self.proj == "cifar10dvs-me":
            return self.generate_cifar10dvs_model_name(args)
        else:
            raise ValueError(
                f"Unknown project: {self.proj}; "
                f"available projects: {self.available_projs}"
            )


class ModelNameParser:

    available_projs = [
        "sequential-cifar10-me",
        "sequential-cifar100-me",
        "cifar10dvs-me",
    ]

    def __init__(self, proj):
        self.proj = proj

    @staticmethod
    def parse_sequential_cifar10_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_sequential_cifar100_model_name(model_name):
        raise NotImplementedError

    @staticmethod
    def parse_cifar10dvs_model_name(model_name):
        raise NotImplementedError

    def parse(self, model_name):
        if self.proj == "sequential-cifar10-me":
            return self.parse_sequential_cifar10_model_name(model_name)
        elif self.proj == "sequential-cifar100-me":
            return self.parse_sequential_cifar100_model_name(model_name)
        elif self.proj == "cifar10dvs-me":
            return self.parse_cifar10dvs_model_name(model_name)
        else:
            raise ValueError(
                f"Unknown project: {self.proj}; "
                f"available projects: {self.available_projs}"
            )
